from torch import nn
import torch
from models.embeddings import Embeddings
from models.lstm import LSTMContextualize
from models.span_embeddings import UnaryScores, RelScores, NerScores
from models.antecedent_score import AntecedentScore
import data_utils
import util
import span_prune_cpp
from models.span_embeddings_wrapper import SpanEmbeddingsWrapper


class Model(nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.config = config
        self.data = data
        self.embeddings = Embeddings(config, data)
        self.lstm = LSTMContextualize(config, data)
        self.span_embeddings_wrapper = SpanEmbeddingsWrapper(config, data, generate_candidates=True)
        self.unary_scores = UnaryScores(config)
        self.antecedent_scores = AntecedentScore(config)
        self.rel_scores = RelScores(config, len(self.data.rel_labels))
        self.ner_scores = NerScores(config, len(self.data.ner_labels))
        self.train_disjoint=True
        # TODO get the length dynamically
        # TODO investigate the no-relation link
        self.rel_embs = nn.Embedding(len(self.data.rel_labels_extended), 500)

        # TODO try without project
        self.emb_projection = nn.Linear(1270, 500)

    def set_train_disjoint(self, value):
        self.train_disjoint = value

    def forward(self, batch):
        # max sentence length in terms of number of words
        max_sentence_length = batch.char_idx.shape[1]

        # Get span representations
        (
            candidate_starts,
            candidate_ends,
            flat_candidate_starts,
            flat_candidate_ends,
            candidate_span_emb,
            candidate_mask,
            flat_candidate_mask,
            head_scores,
            flat_context_emb,
            flat_head_emb,
            batch_word_offset
        ) = self.span_embeddings_wrapper(batch)

        doc_len = flat_context_emb.shape[0]


        num_candidates = candidate_span_emb.shape[0]
        max_num_candidates_per_sentence = candidate_mask.shape[1]

        predict_dict = {
            # [num_sentences, max_mention_width * max_sentence_length] - unavailable indices zeroed out
            "candidate_starts": candidate_starts,
            # [num_sentences, max_mention_width * max_sentence_length] - candidate_starts
            "candidate_ends": candidate_ends
        }

        if head_scores is not None:
            predict_dict["head_scores"] = head_scores

        # [num_sentences, max_num_candidates_per_sentence] - zeros out the padding and arranges the ids
        # in a dense matrix in an absolute way
        candidate_span_ids = util.sparse_to_dense(candidate_mask, num_candidates).type(torch.long)

        # [num_sentences, max_num_candidates_per_sentence]
        spans_log_mask = torch.log(candidate_mask.type(torch.float32))

        # Get entity representations.
        if self.config["relation_weight"] > 0:
            # score candidates
            # [num-candidates]
            flat_candidate_entity_scores = self.unary_scores(candidate_span_emb)

            # Get flat candidate scores in the original shape # [num_sentences, max_num_candidates]
            # give -inf to the padded spans
            candidate_entity_scores = flat_candidate_entity_scores[candidate_span_ids] + spans_log_mask

            # entity_starts = entity_ends = entity_scores(same score as candidate_entity_scores but pruned) = [num_sentences, max_num_ents]
            # num_entities = [num_sentences,]
            # top_entity_indices = [num_sentences, max_num_ents]
            entity_starts, entity_ends, entity_scores, num_entities, top_entity_indices = util.get_batch_topk(
                candidate_starts, candidate_ends, candidate_entity_scores, self.config["entity_ratio"],
                batch.text_len, max_sentence_length, self.config.device, sort_spans=True, enforce_non_crossing=False
            )

            # [num_sentences, max_num_ents]
            # absolute indices (offset added)
            entity_span_indices = util.batch_gather(candidate_span_ids, top_entity_indices)

            # [num_sentences, max_num_ents, emb]
            entity_emb = candidate_span_emb[entity_span_indices]

            entities_mask = util.sequence_mask(num_entities, entity_emb.shape[1]).view(-1)

            top_span_indices_rels = entity_span_indices.view(-1)[entities_mask]

            doc_lens = []
            offset = 0
            doc2idx = {}
            for count, dlen in enumerate(batch.doc_len):
                doc_lens.append(sum(num_entities[offset:offset + dlen]))
                doc2idx[batch.doc_id[offset].item()] = count
                offset += dlen

            # Split span ids
            top_span_indices_rels = list(top_span_indices_rels.split(doc_lens))

        # Get coref representations.
        if self.config["coref_weight"] > 0:
            # score mentions
            # [num-candidates]
            # Score independent from relation pruning
            candidate_mention_scores = self.unary_scores(candidate_span_emb)  # [num_candidates]
            doc_ids = batch.doc_id.unsqueeze(1)

            candidate_doc_ids = torch.masked_select(
                doc_ids.repeat([1, max_num_candidates_per_sentence]).view(-1),
                flat_candidate_mask
            )  # [num_candidates]

            k = torch.floor(torch.tensor(doc_len * self.config["mention_ratio"])).type(torch.int32)

            # Different from the predicted indices from entities, meaning that
            # mention scores are independant from span scores and can result in different spans
            # TODO does the doc_len affect the calculations
            top_mention_indices = span_prune_cpp.extract_spans(
                candidate_mention_scores.unsqueeze(0),
                flat_candidate_starts.unsqueeze(0),
                flat_candidate_ends.unsqueeze(0),
                k.unsqueeze(0),
                doc_len,
                True,
                True
            )  # [1, topk]

            top_mention_indices = torch.squeeze(top_mention_indices).type(torch.int64)  # [k]
            mention_starts = flat_candidate_starts[top_mention_indices]  # [k]
            mention_ends = flat_candidate_ends[top_mention_indices]  # [k]
            mention_scores = candidate_mention_scores[top_mention_indices]  # [k]
            mention_emb = candidate_span_emb[top_mention_indices]  # [k, emb]
            mention_doc_ids = candidate_doc_ids[top_mention_indices]  # [k]

            top_span_indices, mention_idx = self.append_mention_idx(
                top_span_indices_rels,
                doc2idx,
                batch.doc_len,
                top_mention_indices,
                mention_doc_ids
            )

            # ent_coref_lens = [len(item) for item in top_span_indices]

            # Top spans of interest (to be passed to graphwriter)
            top_spans = candidate_span_emb[torch.cat(top_span_indices, 0)]

            num_doc_entities = [len(x) for x in top_span_indices]

            if head_scores is not None:
                predict_dict["coref_head_scores"] = head_scores

            # represents number of mentions per doc
            data = torch.zeros(torch.max(mention_doc_ids) + 1).scatter_add(
                0,
                mention_doc_ids,
                torch.ones_like(mention_doc_ids).type(torch.float32)
            ).type(torch.int64)

            max_mentions_per_doc = torch.max(data)

            # max distance where candidate antecedent can be anticipated
            max_antecedents = torch.min(
                torch.min(torch.tensor(self.config["max_antecedents"]), (k - 1).type(torch.int64)),
                max_mentions_per_doc - 1
            )

            target_indices = torch.arange(k).unsqueeze(1)  # [k, 1]
            antecedent_offsets = (torch.arange(max_antecedents) + 1).unsqueeze(0)  # [1, max_ant]

            raw_antecedents = target_indices - antecedent_offsets  # [k, max_ant]
            antecedents = torch.max(raw_antecedents, torch.tensor(0))  # [k, max_ant]
            target_doc_ids = mention_doc_ids.unsqueeze(1)  # [k, 1]
            antecedent_doc_ids = mention_doc_ids[antecedents]  # [k, max_ant]

            # antecedent has to be greater than 0 and has to belong to the same document
            antecedent_mask = (
                    (target_doc_ids == antecedent_doc_ids)
                    &
                    (raw_antecedents >= 0)
            )  # [k, max_ant]

            antecedent_log_mask = torch.log(antecedent_mask.type(torch.float32))  # [k, max_ant]

            # [k, max_ant], [k, max_ant, emb], [k, max_ant, emb2]
            antecedent_scores, antecedent_emb, pair_emb = self.antecedent_scores(
                mention_emb, mention_scores, antecedents
            )  # [k, max_ant]

            # zero out (out of document range) score and -index scores
            # TODO how to deal with that zerod added record
            antecedent_scores = torch.cat([
                torch.zeros([k, 1]), antecedent_scores + antecedent_log_mask], 1)  # [k, max_ant+1]

        # Get labels.
        if self.config["ner_weight"] + self.config["coref_weight"] > 0:
            gold_ner_labels, gold_coref_cluster_ids = data_utils.get_span_task_labels(
                candidate_starts, candidate_ends, batch,
                max_sentence_length)  # [num_sentences, max_num_candidates]

        if self.config["relation_weight"] > 0:
            # ground truth labels for the top extracted entities
            rel_labels = data_utils.get_relation_labels(
                entity_starts, entity_ends, num_entities, max_sentence_length,
                batch.rel_e1_starts, batch.rel_e1_ends, batch.rel_e2_starts, batch.rel_e2_ends,
                batch.rel_labels, batch.rel_len
            )  # [num_sentences, max_num_ents, max_num_ents]

            # TODO rel mask is needed here to avoid the padding
            rel_scores, rel_loss, rel_loss_mask = self.rel_scores(
                entity_emb,  # [num_sentences, max_num_ents, emb]
                entity_scores,  # [num_sentences, max_num_ents]
                rel_labels,  # [num_sentences, max_num_ents, max_num_ents]
                num_entities
            )  # [num_sentences, max_num_ents, max_num_ents, num_labels]

            # Rearrange scores
            flattened_scores = rel_scores.view(-1, 8)
            masked_flattened_scores = flattened_scores[rel_loss_mask.view(-1)].view(-1)

            # Build graphs per document
            batch.adj, rel_lengths, coref_lengths = self.build_graphs(batch, num_entities, rel_scores,
                                                                      masked_flattened_scores, num_doc_entities,
                                                                      mention_idx, antecedent_scores, antecedent_mask)

            # Rel embeddings
            rel_indices = torch.arange(len(self.data.rel_labels)).repeat(
                sum(num_entities * num_entities)
            )

            batch.top_spans, batch.rels, batch.doc_num_entities = self.prepare_adj_embs(
                top_spans,
                num_doc_entities,
                rel_indices,
                rel_lengths,
                coref_lengths
            )

            print(f"spans size: {batch.top_spans}")

            # TODO figure how to append mentions
            # TODO train without mention
            predict_dict.update({
                # flat candidate scores in the original shape # [num_sentences, max_num_candidates]
                # -inf to the padded spans
                "candidate_entity_scores": candidate_entity_scores,
                #  the top selected entity span starts  [num_sentences, max_selected_spans]
                "entity_starts": entity_starts,
                #  the top selected entity spans ends   [num_sentences, max_selected_spans]
                "entity_ends": entity_ends,
                #  the top selected entity spans score [num_sentences, max_selected_spans]
                "entitiy_scores": entity_scores,
                # the topk calculated by max(ratio*sentence_len, 1)
                "num_entities": num_entities,
                "rel_labels": torch.argmax(rel_scores, -1),  # predicated labels # [num_sentences, num_ents, num_ents]
                "rel_scores": rel_scores  # probability distr score for each label for each pair of entities
            })
        else:
            rel_loss = 0

        # Compute Coref loss.
        if self.config["coref_weight"] > 0:
            flat_cluster_ids = gold_coref_cluster_ids.view(-1)[flat_candidate_mask]  # [num_candidates]
            mention_cluster_ids = flat_cluster_ids[top_mention_indices]  # [k]

            antecedent_cluster_ids = mention_cluster_ids[antecedents]  # [k, max_ant]
            antecedent_cluster_ids += antecedent_log_mask.type(torch.int32)  # [k, max_ant]

            same_cluster_indicator = (
                    antecedent_cluster_ids == mention_cluster_ids.unsqueeze(1)
            )  # [k, max_ant]

            non_dummy_indicator = (mention_cluster_ids > 0).unsqueeze(1)  # [k, 1]
            pairwise_labels = (same_cluster_indicator & non_dummy_indicator)  # [k, max_ant]

            dummy_labels = ~(torch.sum(pairwise_labels, 1, keepdim=True) > 0)  # [k, 1]
            antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1)  # [k, max_ant+1]
            coref_loss = data_utils.get_coref_softmax_loss(antecedent_scores, antecedent_labels)  # [k]
            coref_loss = torch.sum(coref_loss)

            predict_dict.update({
                "candidate_mention_starts": flat_candidate_starts,  # [num_candidates]
                "candidate_mention_ends": flat_candidate_ends,  # [num_candidates]
                "candidate_mention_scores": candidate_mention_scores,  # [num_candidates]
                "mention_starts": mention_starts,  # [k]
                "mention_ends": mention_ends,  # [k]
                "antecedents": antecedents,  # [k, max_ant]
                "antecedent_scores": antecedent_scores,  # [k, max_ant+1]
            })
        else:
            coref_loss = 0

        # [num_sentences, max_num_candidates_per_sentence, 1]
        dummy_scores = torch.zeros_like(candidate_span_ids, dtype=torch.float32).unsqueeze(2)

        if self.config["ner_weight"] > 0:
            # [num_sentences, max_num_candidates, num_labels-1]
            ner_scores, ner_loss = self.ner_scores(
                candidate_span_emb,
                flat_candidate_entity_scores,
                candidate_span_ids,
                spans_log_mask,
                dummy_scores,
                gold_ner_labels,
                candidate_mask
            )

            predict_dict["ner_scores"] = ner_scores

        loss = (self.config["ner_weight"] * ner_loss + (
                self.config["coref_weight"] * coref_loss + self.config["relation_weight"] * rel_loss
        )
                )

        # Build tgt
        # batch.tgt = self.build_tgt(batch, batch_word_offset, top_span_indices, flat_candidate_starts,
        #                            flat_candidate_ends)

        return predict_dict, loss

    def build_graphs(self, batch, num_entities, rel_scores, masked_flattened_scores, num_doc_entities, mention_idx,
                     antecedent_scores, antecedent_mask):
        # Holds adj matrix per document
        adj = []
        offset = 0
        scores_offset = 0
        doc_rel_ind_x = []
        rel_lengths = []
        coref_lengths = []
        coref_doc_offset = 0
        coref_offset = 0
        for idx, i in enumerate(batch.doc_len):

            # Prepare mentions indices
            entity_rel_indx, rel_entity_indy, coref_doc_offset, doc_coref_count = self.arrange_coref_scores_for_document(
                antecedent_mask, mention_idx[idx], coref_doc_offset)

            doc_coref_entities_count = len(mention_idx[idx])

            coref_lengths.append(doc_coref_count)
            doc_entities = num_entities[offset:offset + i]

            entities_scores_x_ind = torch.arange(sum(doc_entities)).repeat_interleave(
                (doc_entities * rel_scores.shape[3]).repeat_interleave(
                    doc_entities.type(torch.int64)
                ).type(torch.int64)
            )

            entities_scores_y_ind = torch.arange(entities_scores_x_ind.shape[0])

            dim_y_without_mentions = (sum(doc_entities * doc_entities) * 8)
            entities_rel_scores_arranged = torch.zeros(num_doc_entities[idx], dim_y_without_mentions + doc_coref_count)

            entities_rel_scores_arranged[
                (entities_scores_x_ind, entities_scores_y_ind)
            ] = masked_flattened_scores[scores_offset:scores_offset + dim_y_without_mentions]

            # coref scores
            entities_rel_scores_arranged[
                (entity_rel_indx,
                 torch.arange(start=dim_y_without_mentions, end=dim_y_without_mentions + doc_coref_count))
            ] = antecedent_scores[coref_offset:coref_offset + doc_coref_entities_count, 1:][
                antecedent_mask[coref_offset:coref_offset + doc_coref_entities_count, :]].view(-1)

            entities_adj = torch.diag(torch.ones(entities_rel_scores_arranged.shape[0] + 1))

            # add global node
            entities_adj[entities_rel_scores_arranged.shape[0]:] = torch.ones(entities_rel_scores_arranged.shape[0] + 1)
            entities_adj[:, entities_rel_scores_arranged.shape[0]] = torch.ones(
                entities_rel_scores_arranged.shape[0] + 1)

            # Upper Graph
            graph_adj_upper = torch.cat([
                entities_adj,
                torch.cat([
                    entities_rel_scores_arranged, torch.zeros(entities_rel_scores_arranged.shape[1]).unsqueeze(0)
                ], 0)
            ], 1)

            # Build Lower Graph
            ent_offset = 0
            for sent_num_ent in doc_entities:
                sent_rel_ind_x = torch.tensor([8]).repeat_interleave((sent_num_ent * sent_num_ent).item())
                doc_rel_ind_x.append(
                    torch.arange(start=ent_offset, end=ent_offset + sent_num_ent.item()).repeat(
                        sent_num_ent).repeat_interleave(sent_rel_ind_x))
                ent_offset += sent_num_ent

            # Get indices for one document
            doc_rel_ind_x_tensor = torch.cat(doc_rel_ind_x, 0)
            doc_rel_ind_y = torch.arange(doc_rel_ind_x_tensor.shape[0])

            # Build matrix for a document
            rel_scores_rearranged = torch.zeros(entities_rel_scores_arranged.shape[1],
                                                entities_rel_scores_arranged.shape[0])

            rel_scores_rearranged[
                (doc_rel_ind_y, doc_rel_ind_x_tensor)
            ] = masked_flattened_scores[scores_offset: scores_offset + dim_y_without_mentions]

            # coref scores
            rel_scores_rearranged[
                (torch.arange(start=dim_y_without_mentions, end=dim_y_without_mentions + doc_coref_count),
                 rel_entity_indy)
            ] = antecedent_scores[coref_offset:coref_offset + doc_coref_entities_count, 1:][
                antecedent_mask[coref_offset:coref_offset + doc_coref_entities_count, :]].view(-1)

            rel_lengths.append(dim_y_without_mentions)

            graph_adj_lower = torch.cat(
                (
                    torch.cat([
                        rel_scores_rearranged,
                        torch.zeros(rel_scores_rearranged.shape[0]).unsqueeze(1)
                    ], 1),
                    torch.diag(torch.ones(rel_scores_rearranged.shape[0]))
                ), 1)

            adj.append(torch.cat((graph_adj_upper, graph_adj_lower), 0))

            # reset values for the next document
            doc_rel_ind_x = []
            offset += i
            scores_offset += dim_y_without_mentions
            coref_offset += doc_coref_entities_count

        return adj, rel_lengths, coref_lengths

    def prepare_adj_embs(self, top_spans, ent_len, rel_indices, rel_lengths, coref_lengths):
        out = self.span_embeddings_wrapper.pad_entities(top_spans, ent_len)

        # list of all relations embs
        rels = self.rel_embs.weight[rel_indices].split(rel_lengths)

        # coref rels
        coref_rels = self.rel_embs.weight[
            torch.tensor(
                self.data.rel_labels_extended['MERGE']
            ).repeat(sum(coref_lengths))
        ].split(coref_lengths)

        rels_list = []

        for idx, rel in enumerate(rels):
            rels_list.append(
                torch.cat(
                    (
                        torch.cat((self.rel_embs.weight[self.data.rel_labels_extended['ROOT']].unsqueeze(0), rel), 0),
                        coref_rels[idx]
                    ), 0)
            )

        return out, rels_list, torch.tensor(ent_len)

    def append_mention_idx(self, top_span_indices, doc2idx, doc_len, top_mention_indices, mention_doc_ids):
        mention_idx = [[] for i in range(len(doc_len))]
        # mention_new_entities = [0 for i in range(len(doc_len))]

        top_span_indices = list(top_span_indices)

        for idx, mention_doc_id in enumerate(mention_doc_ids):
            doc_idx = doc2idx[mention_doc_id.item()]
            mention_span_idx = top_mention_indices[idx]

            span_idx = util.index(top_span_indices[doc_idx], mention_span_idx.item())

            if span_idx:
                mention_idx[doc_idx].append(span_idx)
            else:
                top_span_indices[doc_idx] = torch.cat((top_span_indices[doc_idx], torch.tensor([mention_span_idx])))
                mention_idx[doc_idx].append(top_span_indices[doc_idx].shape[0] - 1)
                # mention_new_entities[doc_idx] += 1
        return top_span_indices, mention_idx

    def arrange_coref_scores_for_document(self, antecedent_mask, mention_idx, doc_offset):
        """
        Arranges coref scores into indices ready to be integrated in the adjcancy matrix
        """
        doc_coref_count = 0
        entity_rel_indx = []
        rel_entity_indy = []
        for idx, candidate in enumerate(mention_idx):
            rels = len(antecedent_mask[doc_offset].nonzero())
            entity_rel_indx += [item for item in [candidate] for i in range(rels)]
            rel_entity_indy += [mention_idx[i] for i in range(rels)]
            doc_coref_count += rels
            doc_offset += 1

        return entity_rel_indx, rel_entity_indy, doc_offset, doc_coref_count

    # TODO
    def build_tgt(self, batch, batch_word_offset, top_span_indices, flat_candidate_starts, flat_candidate_ends):
        # combine all document spans into one tensor
        top_span_indices_combined = torch.cat(top_span_indices, 0)
        # Get span boundaries (shifted)
        span_starts = flat_candidate_starts[top_span_indices_combined]
        span_ends = flat_candidate_ends[top_span_indices_combined]

        span_boundries = {}
        for idx, span_idx in enumerate(top_span_indices_combined):
            span_boundries[f"{span_starts[idx]}-{span_ends[idx]}"] = span_idx.item()

        # Build dict start-end => index
        tgts_entities = [[] for i in range(len(batch.ner_len))]
        for idx, ner_count in enumerate(batch.ner_len):
            # get gold boundaris offseted for each sentence
            gold_starts = batch.ner_starts[idx][:ner_count] + batch_word_offset[idx]
            gold_ends = batch.ner_ends[idx][:ner_count] + batch_word_offset[idx]

            # match against span indices to get index if exists
            for span_idx, span in enumerate(gold_starts):
                key = f"{gold_starts[span_idx]}-{gold_ends[span_idx]}"
                if key in span_boundries:
                    tgts_entities[idx].append(
                        (
                            batch.ner_starts[idx][span_idx].type(torch.int32).item(),
                            batch.ner_ends[idx][span_idx].type(torch.int32).item(),
                            batch.ner_labels[idx][span_idx].type(torch.int32).item(),
                            span_boundries[key]
                        )
                    )

        # If index exists, replace tgt that match these boundries with the label
        out_texts = []
        for sent_idx, sentence in enumerate(batch.tokens):
            out_text = []
            current_ent = None
            sentence = sentence[:batch.text_len[sent_idx]]
            ner_pointer = 0
            if tgts_entities[sent_idx]:
                current_ent = (
                    tgts_entities[sent_idx][ner_pointer][0],
                    tgts_entities[sent_idx][ner_pointer][1],
                    f"<{self.data.ner_labels_inv[tgts_entities[sent_idx][ner_pointer][2]].replace(' ', '').lower()}_{tgts_entities[sent_idx][ner_pointer][3]}>"
                )

            for idx, word in enumerate(sentence):
                # Update current ent if necessary
                if current_ent and idx > current_ent[1]:
                    ner_pointer += 1

                    if len(tgts_entities[sent_idx]) > ner_pointer:
                        current_ent = (
                            tgts_entities[sent_idx][ner_pointer][0],
                            tgts_entities[sent_idx][ner_pointer][1],
                            f"<{self.data.ner_labels_inv[tgts_entities[sent_idx][ner_pointer][2]].replace(' ', '').lower()}_{tgts_entities[sent_idx][ner_pointer][3]}>"
                        )
                    else:
                        current_ent = None

                if not current_ent or idx < current_ent[0]:
                    out_text.append(word)
                    continue

                if idx >= current_ent[0] and idx <= current_ent[1]:
                    # If this marks the end word of the span, append the type here
                    if idx == current_ent[1]:
                        out_text.append(current_ent[2])
                        continue
            out_texts.append(out_text)

        print(out_texts)

            # TODO merge document and build vocab




