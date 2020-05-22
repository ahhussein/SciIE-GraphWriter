from torch import nn
import torch
from models.embeddings import Embeddings
from models.lstm import LSTMContextualize
from models.span_embeddings import SpanEmbeddings, UnaryScores
import data_utils
import util


class Model(nn.Module):
    def __init__(self, config, data, is_training=1):
        super().__init__()
        self.config = config
        self.data = data
        self.embeddings = Embeddings(config, data, is_training)
        self.lstm = LSTMContextualize(config, data, is_training)
        self.span_embeddings = SpanEmbeddings(config, data, is_training)
        self.unary_scores = UnaryScores(config, is_training)

    def forward(self, batch):
        max_sentence_length = batch.char_idx.shape[1]

        # context_emb = [num_sentences, max_sentence_length, emb1]
        # head_emb    = [num_sentences, max_sentence_length, emb2]
        context_emb, head_emb, lm_weights, lm_scaling = self.embeddings(batch)

        # [max-num_sentences, max-sentence-lenth, num-dir=2 * hz]
        context_outputs = self.lstm(context_emb)

        # TODO test lines
        # print('context_emb - lstm out')
        # print(context_outputs.shape)
        # print(context_outputs)

        # [num_sentences, max_mention_width * max_sentence_length]
        candidate_starts, candidate_ends, candidate_mask = data_utils.get_span_candidates(
            batch.text_len,
            batch.char_idx.shape[1],
            self.config['max_arg_width']
        )

        # # TODO test lines
        # print('candidate start shape')
        # print(candidate_starts.shape)
        #
        # print('candidate mask shape')
        # print(candidate_mask.shape)
        #
        # print('candidate start')
        # print(candidate_starts)
        # print('candidate ends')
        # print(candidate_ends)
        # print('mask')
        # print(candidate_mask)

        flat_candidate_mask = candidate_mask.view(-1) # [num_sentences * max_mention_width * max_sentence_length]

        # Perform exclusive cum sum
        batch_word_offset = torch.cumsum(batch.text_len, 0).roll(1).unsqueeze(1)
        batch_word_offset[0] = 0 # [num_sentences, 1]
        # print('cum sum offsets')
        # print(batch.text_len)
        # print(batch_word_offset)

        # broadcast offset shifting to all sentences, and apply mask select
        # [num_candidates]
        flat_candidate_starts = torch.masked_select(
            (candidate_starts + batch_word_offset).view(-1),
            flat_candidate_mask
        )
        # print("flat candidate start")
        # print(flat_candidate_starts)

        # [num_candidates], words offsets added to remove sentence boundaries
        flat_candidate_ends = torch.masked_select(
            (candidate_ends + batch_word_offset).view(-1),
            flat_candidate_mask
        )

        # [num_sentences, max_sentence_length]
        text_len_mask = util.sequence_mask(batch.text_len, max_sentence_length)

        # [num_words, emb], padding removed
        flat_context_outputs = util.flatten_emb_by_sentence(context_outputs, text_len_mask)
        # [num_words, emb]
        flat_head_emb = util.flatten_emb_by_sentence(head_emb, text_len_mask)

        # TODO ensure that sample are not cross documents
        doc_len = flat_context_outputs.shape[0]

        # [num_candidates, emb], [num_candidates, max_span_width, emb], [num_candidates, max_span_width]
        candidate_span_emb, head_scores, span_text_emb, span_indices, span_indices_log_mask = self.span_embeddings(
            flat_head_emb,
            flat_context_outputs,
            flat_candidate_starts,
            flat_candidate_ends
        )

        num_candidates = candidate_span_emb[0]
        max_num_candidates_per_sentence = candidate_mask[1]


        # TODO dense to sparse

        # [num_sentences, max_num_candidates_per_sentence]
        span_log_mask = torch.log(candidate_mask.type(torch.float32))

        # TODO what is that used for?
        predict_dict = {"candidate_starts": candidate_starts, "candidate_ends": candidate_ends}

        if head_scores is not None:
            predict_dict["head_scores"] = head_scores

        candidate_span_ids = util.sparse_to_dense(candidate_mask, candidate_span_emb.shape[0])

        # [num_sentences, max_num_candidates]
        spans_log_mask = torch.log(candidate_mask.type(torch.float32))

        # Get entity representations.
        if self.config["relation_weight"] > 0:
            # [num_sentences, num_spans]
            flat_candidate_entity_scores = self.unary_scores(candidate_span_emb)

            # Adds -inf to the padding entries (spans) # [num_sentences, max_num_candidates]
            candidate_entity_scores = flat_candidate_entity_scores[candidate_span_ids] + spans_log_mask



























