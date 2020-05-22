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








