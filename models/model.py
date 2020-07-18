from torch import nn
import torch
from models.embeddings import Embeddings
from models.lstm import LSTMContextualize
from models.span_embeddings import SpanEmbeddings, UnaryScores, RelScores, NerScores
from models.antecedent_score import AntecedentScore
import data_utils
import util
import span_prune_cpp

class Model(nn.Module):
    def __init__(self, config, data, is_training=1):
        super().__init__()
        self.config = config
        self.data = data
        self.embeddings = Embeddings(config, data, is_training)
        self.lstm = LSTMContextualize(config, data, is_training)
        self.span_embeddings = SpanEmbeddings(config, data, is_training)
        self.unary_scores = UnaryScores(config, is_training)
        self.antecedent_scores = AntecedentScore(config, is_training)
        self.rel_scores = RelScores(config, len(self.data.rel_labels), is_training)
        self.ner_scores = NerScores(config, len(self.data.ner_labels), is_training)

        # TODO get the length dynamically
        # TODO investigate the no-relation link
        self.rel_embs = nn.Embedding(len(self.data.rel_labels) + 1, 500)
        self.emb_projection = nn.Linear(1270, 500)

    def forward(self, batch):
        # max sentence length in terms of number of words
        max_sentence_length = batch.char_idx.shape[1]

        # context_emb = [num_sentences, max_sentence_length, emb1]
        # head_emb    = [num_sentences, max_sentence_length, emb2]
        context_emb, head_emb, lm_weights, lm_scaling = self.embeddings(batch)

        # [num_sentences, max-sentence-length, num-dir=2 * hz]
        context_outputs = self.lstm(context_emb)

        # [num_sentences, max_mention_width * max_sentence_length]
        candidate_starts, candidate_ends, candidate_mask = data_utils.get_span_candidates(
            batch.text_len,
            batch.char_idx.shape[1],
            self.config['max_arg_width']
        )

        flat_candidate_mask = candidate_mask.view(-1) # [num_sentences * max_mention_width * max_sentence_length]

        # Perform exclusive cum sum
        batch_word_offset = torch.cumsum(batch.text_len, 0).roll(1).unsqueeze(1)
        batch_word_offset[0] = 0 # [num_sentences, 1]

        # broadcast offset shifting to all sentences, and apply mask select
        # Sentence shifting will ensure that the word gets matched to the corresponding
        # embedding in the `flat_context_emb` and `flat_head_emb`
        # [num_candidates]
        flat_candidate_starts = torch.masked_select(
            (candidate_starts + batch_word_offset).view(-1),
            flat_candidate_mask
        )

        # [num_candidates], words offsets added to remove sentence boundaries
        flat_candidate_ends = torch.masked_select(
            (candidate_ends + batch_word_offset).view(-1),
            flat_candidate_mask
        )

        # [num_sentences, max_sentence_length]
        text_len_mask = util.sequence_mask(batch.text_len, max_sentence_length)

        # [num_words, emb], padding removed
        flat_context_emb = util.flatten_emb_by_sentence(context_outputs, text_len_mask)

        # [num_words, emb]
        flat_head_emb = util.flatten_emb_by_sentence(head_emb, text_len_mask)

        # doc len in terms of words in all samples in a batch
        # TODO will that be effected if doc_len is longer?
        # doc len in terms of words in document
        doc_len = flat_context_emb.shape[0]

        # [num_candidates, emb], [num_candidates, max_span_width, emb], [num_candidates, max_span_width]
        # candidate_span_emb is the concat of candidate-start word emb, candidate-end word emb and width emb
        # as well span head emb
        candidate_span_emb, head_scores, span_text_emb, span_indices, span_indices_log_mask = self.span_embeddings(
            flat_head_emb,
            flat_context_emb,
            flat_candidate_starts,
            flat_candidate_ends
        )

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
                candidate_starts, candidate_ends, candidate_entity_scores, self.config["entity_ratio"] * 0.25, # TODO
                batch.text_len, max_sentence_length, sort_spans=True, enforce_non_crossing=False
            )

            # [num_sentences, max_num_ents]
            # absolute indices (offset added)
            entity_span_indices = util.batch_gather(candidate_span_ids, top_entity_indices)

            # [num_sentences, max_num_ents, emb]
            entity_emb = candidate_span_emb[entity_span_indices]

            entities_mask = util.sequence_mask(num_entities, entity_emb.shape[1]).view(-1)

            # Top spans of interest (to be passed to graphwriter)
            top_spans = entity_emb.view(entity_emb.shape[0] * entity_emb.shape[1], -1)[entities_mask]

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

            target_indices = torch.arange(k).unsqueeze(1) # [k, 1]
            antecedent_offsets = (torch.arange(max_antecedents) + 1).unsqueeze(0) # [1, max_ant]

            raw_antecedents = target_indices - antecedent_offsets  # [k, max_ant]
            antecedents = torch.max(raw_antecedents, torch.tensor(0))  # [k, max_ant]
            target_doc_ids = mention_doc_ids.unsqueeze(1)  # [k, 1]
            antecedent_doc_ids = mention_doc_ids[antecedents]  # [k, max_ant]

            # antecedent has to be greater than 0 and has to belong to the same document
            antecedent_mask = (
                (target_doc_ids == antecedent_doc_ids)
                &
                (raw_antecedents >= 0)
            ) # [k, max_ant]

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
            #ground truth labels for the top extracted entities
            rel_labels = data_utils.get_relation_labels(
                entity_starts, entity_ends, num_entities, max_sentence_length,
                batch.rel_e1_starts, batch.rel_e1_ends, batch.rel_e2_starts, batch.rel_e2_ends,
                batch.rel_labels, batch.rel_len
            )  # [num_sentences, max_num_ents, max_num_ents]

            # TODO rel mask is needed here to avoid the padding
            rel_scores, rel_loss, rel_loss_mask = self.rel_scores(
                entity_emb,   # [num_sentences, max_num_ents, emb]
                entity_scores,  # [num_sentences, max_num_ents]
                rel_labels, # [num_sentences, max_num_ents, max_num_ents]
                num_entities
            )  # [num_sentences, max_num_ents, max_num_ents, num_labels]

            # Rearrange scores
            flattened_scores = rel_scores.view(-1, 8)
            masked_flattened_scores = flattened_scores[rel_loss_mask.view(-1)].view(-1)

            # Build graphs per document
            batch.adj, rel_lengths = self.build_graphs(batch, num_entities, rel_scores, masked_flattened_scores)

            # Rel embeddings
            rel_indices = torch.arange(len(self.data.rel_labels)).repeat(
                sum(num_entities * num_entities)
            )

            batch.top_spans, batch.rels, batch.doc_num_entities = self.prepare_adj_embs(top_spans, num_entities, rel_indices, rel_lengths, batch.doc_len)

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
                "rel_labels": torch.argmax(rel_scores, -1), # predicated labels # [num_sentences, num_ents, num_ents]
                "rel_scores": rel_scores # probability distr score for each label for each pair of entities
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

            dummy_labels = ~(torch.sum(pairwise_labels, 1, keepdim=True) > 0)# [k, 1]
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

        return predict_dict, loss

    def build_graphs(self, batch, num_entities, rel_scores, masked_flattened_scores):
        # Holds adj matrix per document
        adj = []
        offset = 0
        scores_offset = 0
        doc_rel_ind_x = []
        rel_lengths = []

        for i in batch.doc_len:
            doc_entities = num_entities[offset:offset + i]

            entities_scores_x_ind = torch.arange(sum(doc_entities)).repeat_interleave(
                (doc_entities * rel_scores.shape[3]).repeat_interleave(
                    doc_entities.type(torch.int64)
                ).type(torch.int64)
            )

            entities_scores_y_ind = torch.arange(entities_scores_x_ind.shape[0])

            entities_rel_scores_arranged = torch.zeros(sum(doc_entities), sum(doc_entities * doc_entities) * 8)


            entities_rel_scores_arranged[
                (entities_scores_x_ind, entities_scores_y_ind)
            ] = masked_flattened_scores[scores_offset:scores_offset +entities_rel_scores_arranged.shape[1]]

            entities_adj = torch.diag(torch.ones(entities_rel_scores_arranged.shape[0]+1))

            # add global node
            entities_adj[entities_rel_scores_arranged.shape[0]:] = torch.ones(entities_rel_scores_arranged.shape[0]+1)
            entities_adj[:,entities_rel_scores_arranged.shape[0]] = torch.ones(entities_rel_scores_arranged.shape[0]+1)

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
                    torch.arange(start=ent_offset, end=ent_offset+sent_num_ent.item()).repeat(sent_num_ent).repeat_interleave(sent_rel_ind_x))
                ent_offset += sent_num_ent


            # Get indices for one document
            doc_rel_ind_x_tensor = torch.cat(doc_rel_ind_x, 0)
            doc_rel_ind_y = torch.arange(doc_rel_ind_x_tensor.shape[0])

            rel_lengths.append(doc_rel_ind_x_tensor.shape[0])

            # Build matrix for a document
            rel_scores_rearranged = torch.zeros(entities_rel_scores_arranged.shape[1], sum(doc_entities))

            rel_scores_rearranged[
                (doc_rel_ind_y, doc_rel_ind_x_tensor)
            ] = masked_flattened_scores[scores_offset: scores_offset + rel_scores_rearranged.shape[0]]

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
            scores_offset += entities_rel_scores_arranged.shape[1]

        return adj, rel_lengths

    def prepare_adj_embs(self, top_spans, num_entities, rel_indices, rel_lengths, doc_len):
        # Project entity embs to lower space
        top_spans = self.emb_projection(top_spans)

        # Get document lengths
        offset = 0
        ent_len = []
        for i in doc_len:
            ent_len.append(sum(num_entities[offset: i + offset]))
            offset += i

        # Max # entity per sample.
        m = max(ent_len)

        # list of all entities matrics padded to the max entity length
        encs = [self.pad(x, m) for x in top_spans.split(ent_len)]

        # Stack them to end up with 32 * maxlen_of_entities * hidden size
        out = torch.stack(encs, 0)

        # list of all relations embs
        rels = self.rel_embs.weight[rel_indices].split(rel_lengths)

        rels_list = []

        for rel in rels:
            rels_list.append(torch.cat((self.rel_embs.weight[len(self.data.rel_labels)].unsqueeze(0), rel), 0))

        return out, rels_list, torch.tensor(ent_len)



    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])


