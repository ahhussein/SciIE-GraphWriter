from torch import nn, dtype
import torch
import util

class SpanEmbeddings(nn.Module):
    def __init__(self, config, data, is_training=1):
        super().__init__()
        self.config = config
        self.data = data
        self.embeddings = nn.init.xavier_uniform_(
            torch.empty(self.config['max_arg_width'], self.config['feature_size'])
        )
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(1- is_training * self.config['dropout_rate'])
        self.ffnn = nn.Linear(
            config['contextualization_size'] * config['contextualization_layers'] * 2,
            self.config['num_attention_heads']
        )
        torch.nn.init.xavier_uniform_(self.ffnn.weight)

    def forward(self, head_emb, context_outputs, span_starts, span_ends):
        """Compute span representation shared across tasks.
        For each span it computes embs for start word, end word, and the span width

        Args:
          head_emb: Tensor of [num_words, emb]
          context_outputs: Tensor of [num_words, emb]
          span_starts: [num_spans]
          span_ends: [num_spans]
        """
        text_length = context_outputs.shape[0]

        # [num_words, emb]
        span_start_emb = context_outputs[span_starts]
        # [num_words, emb]
        span_end_emb = context_outputs[span_ends]
        span_emb_list = [span_start_emb, span_end_emb]

        # [num-spans]
        span_widths = 1 + span_ends - span_starts
        max_arg_width = self.config['max_arg_width']

        if self.config["use_features"]:
            span_width_index = span_widths - 1
            # [num_spans, emb]
            # Embeddings for widths [#num_spans, feature_size]
            span_width_emb = self.dropout(self.embeddings[span_width_index])
            span_emb_list.append(span_width_emb)

        head_scores = None
        span_text_emb = None
        span_indices = None
        span_indices_log_mask = None

        # TODO read literature head embeddings
        if self.config['model_heads']:
            # [num_spans, max_span_width]
            span_indices = torch.min(
                torch.arange(max_arg_width).unsqueeze(0) + span_starts.unsqueeze(1),
                torch.tensor([text_length - 1])
            )

            span_text_emb = head_emb[span_indices]

            # [num_spans, max_arg_width]
            span_indices_log_mask = torch.log(
                util.sequence_mask(
                    span_widths,
                    max_arg_width,
                    dtype=torch.float32)
            )

            head_scores = self.ffnn(context_outputs)

            # [num_spans, max_arg_width, num_heads]
            span_attention = self.softmax(
                head_scores[span_indices] + span_indices_log_mask.unsqueeze(2)
            )

            # [num_spans, emb]
            span_head_emb = torch.sum(span_attention * span_text_emb, 1)

            span_emb_list.append(span_head_emb)

        # [num_spans, emb]
        span_emb = torch.cat(span_emb_list, 1)

        return span_emb, head_scores, span_text_emb, span_indices, span_indices_log_mask

class UnaryScores(nn.Module):
    def __init__(self, config, is_training=1):
        super().__init__()
        self.config = config
        self.input = nn.Linear(
            1270, #TODO
            self.config["ffnn_size"]
        )

        self.hidden1 = nn.Linear(
            self.config["ffnn_size"],
            self.config["ffnn_size"]
        )

        self.output = nn.Linear(
            self.config["ffnn_size"],
            1
        )

        self.dropout = nn.Dropout(1 - is_training * self.config['dropout_rate'])
        torch.nn.init.xavier_uniform_(self.input.weight)
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.xavier_uniform_(self.output.weight)

    def forward(self, candidate_span_emb):
        x = self.output(
            self.hidden1(
                self.input(candidate_span_emb)
            )
        )

        scores = self.dropout(x)

        # [num_sentences, num_spans] or [k]
        return torch.squeeze(scores)


class RelScores(nn.Module):
    def __init__(self, config, num_labels, is_training=1):
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        self.input = nn.Linear(
            3810, #TODO
            self.config["ffnn_size"]
        )

        self.hidden1 = nn.Linear(
            self.config["ffnn_size"],
            self.config["ffnn_size"]
        )

        self.output = nn.Linear(
            self.config["ffnn_size"],
            num_labels - 1
        )

        self.loss = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(1 - is_training * self.config['dropout_rate'])
        torch.nn.init.xavier_uniform_(self.input.weight)
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.xavier_uniform_(self.output.weight)

    def forward(self, entity_emb, entity_scores, rel_labels, num_predicted_entities):
        num_sentences = entity_emb.shape[0]
        num_entities = entity_emb.shape[1]

        e1_emb_expanded = entity_emb.unsqueeze(2)  # [num_sents, num_ents, 1, emb]
        e2_emb_expanded = entity_emb.unsqueeze(1)  # [num_sents, 1, num_ents, emb]
        e1_emb_tiled = e1_emb_expanded.repeat([1, 1, num_entities, 1])  # [num_sents, num_ents, num_ents, emb]
        e2_emb_tiled = e2_emb_expanded.repeat([1, num_entities, 1, 1])  # [num_sents, num_ents, num_ents, emb]

        similarity_emb = e1_emb_expanded * e2_emb_expanded  # [num_sents, num_ents, num_ents, emb]
        pair_emb_list = [e1_emb_tiled, e2_emb_tiled, similarity_emb]
        pair_emb = torch.cat(pair_emb_list, 3)  # [num_sentences, num_ents, num_ents, emb]
        pair_emb_size = pair_emb.shape[3]
        flat_pair_emb = pair_emb.view([num_sentences * num_entities * num_entities, pair_emb_size])

        x = self.output(
            self.hidden1(
                self.input(flat_pair_emb)
            )
        )

        scores = self.dropout(x)

        # [num_sentences * num_ents * num_ents, 1]
        flat_rel_scores = torch.squeeze(scores)

        rel_scores = flat_rel_scores.view([num_sentences, num_entities, num_entities, self.num_labels - 1])
        # [num_sentences, ents, max_num_ents, num_labels-1]
        rel_scores += entity_scores.unsqueeze(2).unsqueeze(3) + entity_scores.unsqueeze(1).unsqueeze(3)

        dummy_scores = torch.zeros([num_sentences, num_entities, num_entities, 1], dtype=torch.float32)
        # [num_sentences, max_num_ents, max_num_ents, num_labels]
        rel_scores = torch.cat([dummy_scores, rel_scores], 3)

        max_num_entities = rel_scores.shape[1]
        num_labels = rel_scores.shape[3]
        entities_mask = util.sequence_mask(num_predicted_entities, max_num_entities)  # [num_sentences, max_num_entities]

        rel_loss_mask = (
            entities_mask.unsqueeze(2) # [num_sentences, max_num_entities, 1]
            &
            entities_mask.unsqueeze(1) # [num_sentences, 1, max_num_entities]
        )  # [num_sentences, max_num_entities, max_num_entities]

        loss = self.loss(rel_scores.view([-1, num_labels]), rel_labels.reshape(-1))

        loss = torch.masked_select(loss, rel_loss_mask.view(-1))
        return rel_scores, torch.sum(loss)

class NerScores(nn.Module):
    def __init__(self, config, num_labels, is_training=1):
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        self.input = nn.Linear(
            1270, #TODO
            self.config["ffnn_size"]
        )

        self.hidden1 = nn.Linear(
            self.config["ffnn_size"],
            self.config["ffnn_size"]
        )

        self.output = nn.Linear(
            self.config["ffnn_size"],
            num_labels - 1
        )

        self.loss = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(1 - is_training * self.config['dropout_rate'])
        torch.nn.init.xavier_uniform_(self.input.weight)
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.xavier_uniform_(self.output.weight)

    def forward(self, candidate_span_emb, flat_candidate_entity_scores,
                candidate_span_ids, spans_log_mask, dummy_scores,
                gold_ner_labels, candidate_mask
    ):
        x = self.output(
            self.hidden1(
                self.input(candidate_span_emb)
            )
        )

        flat_ner_scores = torch.squeeze(self.dropout(x))

        if self.config["span_score_weight"] > 0:
            flat_ner_scores += (
                    self.config["span_score_weight"] * flat_candidate_entity_scores.unsqueeze(1)
            )

        # [num_sentences, max_num_candidates, num_labels-1]
        ner_scores = flat_ner_scores[candidate_span_ids] + spans_log_mask.unsqueeze(2)

        ner_scores = torch.cat([dummy_scores, ner_scores], 2)  # [num_sentences, max_num_candidates, num_labels]

        num_labels = ner_scores.shape[2]

        #TODO urgent, is that the right loss
        ner_loss = self.loss(ner_scores.view([-1, num_labels]), gold_ner_labels.type(torch.int64).view(-1))

        ner_loss = torch.sum(torch.masked_select(ner_loss, candidate_mask.view(-1)))

        return flat_ner_scores, ner_loss
















