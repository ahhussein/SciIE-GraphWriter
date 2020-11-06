from torch import nn
import torch
import util
from models.embeddings import Embeddings
from models.lstm import LSTMContextualize
import data_utils
from models.span_embeddings import SpanEmbeddings

class VertexEmbeddings(nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.embeddings = Embeddings(config, data)
        self.lstm = LSTMContextualize(config, data)
        self.span_embeddings = SpanEmbeddings(config, data)
        self.config = config
        self.emb_projection = nn.Linear(1270, 500)
        torch.nn.init.xavier_uniform_(self.emb_projection.weight)
        self.rel_embs = nn.Embedding(2 * len(data.rel_labels_extended) - 3, 500)

    def forward(self, batch, generate_candiadtes=True):
        # max sentence length in terms of number of words
        max_sentence_length = batch.char_idx.shape[1]

        # context_emb = [num_sentences, max_sentence_length, emb1]
        # head_emb    = [num_sentences, max_sentence_length, emb2]
        context_emb, head_emb, lm_weights, lm_scaling = self.embeddings(batch)

        # [num_sentences, max-sentence-length, num-dir=2 * hz]
        context_outputs = self.lstm(context_emb)

        # [num_sentences, max_sentence_length]
        # Get gold entities
        if not generate_candiadtes:
            candidate_starts = batch.ner_starts
            candidate_ends = batch.ner_ends
            candidate_mask = util.sequence_mask(batch.ner_len, batch.ner_starts.shape[1])
        else:
            candidate_starts, candidate_ends, candidate_mask = data_utils.get_span_candidates(
                batch.text_len,
                batch.char_idx.shape[1],
                self.config['max_arg_width']
            )

        flat_candidate_mask = candidate_mask.view(-1)  # [num_sentences * max_mention_width * max_sentence_length]

        # Perform exclusive cum sum
        batch_word_offset = torch.cumsum(batch.text_len, 0).roll(1).unsqueeze(1)
        batch_word_offset[0] = 0  # [num_sentences, 1]

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

        # [num_candidates, emb], [num_candidates, max_span_width, emb], [num_candidates, max_span_width]
        # candidate_span_emb is the concat of candidate-start word emb, candidate-end word emb and width emb
        # as well span head emb
        candidate_span_emb, head_scores, span_text_emb, span_indices, span_indices_log_mask = self.span_embeddings(
            flat_head_emb,
            flat_context_emb,
            flat_candidate_starts,
            flat_candidate_ends
        )
        # TODO check projection important for graph to work
        # Project entity embs to lower space
        candidate_span_emb = self.emb_projection(candidate_span_emb)

        return (
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
        )

    def pad_entities(self, spans, lens):
        # Max # entity per sample.
        m = max(lens)

        # list of all entities matrics padded to the max entity length
        encs = [self.pad(x, m) for x in spans.split(lens)]

        # Stack them to end up with 32 * maxlen_of_entities * hidden size
        return torch.stack(encs, 0)

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])









