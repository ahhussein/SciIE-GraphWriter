import pyhocon
import os
import errno
import torch
import span_prune_cpp
import sys

def get_config(filename):
    return pyhocon.ConfigFactory.parse_file(filename)


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def print_config(config):
    print(pyhocon.HOCONConverter.convert(config, "hocon"))


def set_gpus(*gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def flatten(l):
    return [item for sublist in l for item in sublist]


def print_ner(sent_example):
    for ner in sent_example['ner']:
        print_range(
            sent_example['sentence'],
            ner[0] - sent_example['word_offset'],
            ner[1] - sent_example['word_offset']
        )
        print(f" => {ner[2]}")


def print_relations(sent_example):
    for rel in sent_example['relations']:
        # Print first entity
        print_range(
            sent_example['sentence'],
            rel[0] - sent_example['word_offset'],
            rel[1] - sent_example['word_offset']
        )

        print(" == ", end='')

        # Print second entity
        print_range(
            sent_example['sentence'],
            rel[2] - sent_example['word_offset'],
            rel[3] - sent_example['word_offset']
        )

        print(" => ", end='')

        # Print relation type
        print(rel[4])


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()

    row_vector = torch.arange(0, maxlen)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask = mask.type(dtype)
    return mask


def flatten_emb(emb):
    """
    Flatten embeddings
    Params:
      emb [num-sentences, max-sentence-length, emb]
    Returns:
      [num-sentences * max-sentence-length, emb]
    """
    num_sentences = emb.shape[0]
    max_sentence_length = emb.shape[1]
    emb_rank = len(emb.shape)

    if emb_rank == 2:
        flattened_emb = emb.view(num_sentences * max_sentence_length)
    elif emb_rank == 3:
        flattened_emb = emb.reshape(num_sentences * max_sentence_length, -1)
    else:
        raise ValueError("Unsupported rank: {}".format(emb_rank))

    return flattened_emb


def flatten_emb_by_sentence(emb, text_len_mask):
    flattened_emb = flatten_emb(emb)

    return flattened_emb[text_len_mask.view(-1)]
    #     flattened_emb,
    #     text_len_mask.view(-1).unsqueeze(1).repeat([1, flattened_emb[1]])
    # )


def sparse_to_dense(candidate_mask, num_candidates):
    candidate_mask = candidate_mask > 0
    num_sentences = candidate_mask.shape[0]
    max_num_candidates_per_sentence = candidate_mask.shape[1]

    output = torch.zeros((num_sentences, max_num_candidates_per_sentence), dtype=torch.int32)
    sparse_indices = candidate_mask.nonzero()
    sparse_values = torch.arange(num_candidates)
    for i in range(len(sparse_indices)):
        output[tuple(sparse_indices[i])] = sparse_values[i]

    return output


def get_batch_topk(
        candidate_starts,
        candidate_ends,
        candidate_scores,
        topk_ratio,
        text_len,
        max_sentence_length,
        sort_spans=False,
        enforce_non_crossing=True
):
    """
      Args:
        candidate_starts: [num_sentences, max_num_candidates]
        candidate_mask: [num_sentences, max_num_candidates]
        topk_ratio: A float number.
        text_len: [num_sentences,]
        max_sentence_length:
        enforce_non_crossing: Use regular top-k op if set to False.
     """
    num_sentences = candidate_starts.shape[0]

    # [num_sentences]
    topk = torch.max(
        torch.floor(text_len.type(torch.float32) * topk_ratio).type(torch.int32),
        torch.ones(num_sentences, dtype=torch.int32)
    )

    # [num_sentences, max_num_predictions]
    # print(candidate_scores)
    # print(candidate_scores.shape)
    # print(candidate_starts)
    # print(candidate_starts.shape)
    # print(candidate_ends)
    # print(candidate_ends.shape)
    # print(topk)
    # print(topk.shape)
    # print(max_sentence_length)
    min = -sys.maxsize - 1
    candidate_scores[candidate_scores == float('-inf')] = min
    predicted_indices = span_prune_cpp.extract_spans(
        candidate_scores,
        candidate_starts,
        candidate_ends,
        topk,
        max_sentence_length,
        False,
        True
    )

    predicted_indices = predicted_indices.type(torch.int64)
    predicted_starts = batch_gather(candidate_starts, predicted_indices)  # [num_sentences, max_num_predictions]
    predicted_ends = batch_gather(candidate_ends, predicted_indices)  # [num_sentences, max_num_predictions]
    predicted_scores = batch_gather(candidate_scores, predicted_indices)  # [num_sentences, max_num_predictions]

    return predicted_starts, predicted_ends, predicted_scores, topk, predicted_indices


def batch_gather(emb, indices):
    """
    Args:
      emb: Shape of TODO
      indices: Shape of [num_sentences, k, (l)]
    """
    num_sentences = emb.shape[0]
    max_sentence_length = emb.shape[1]
    flattened_emb = flatten_emb(emb)  # [num_sentences * max_sentence_length, emb]
    offset = (torch.arange(num_sentences) * max_sentence_length).unsqueeze(1)  # [num_sentences, 1]
    if len(indices.shape) == 3:
        offset = offset.unsqueeze(2)  # [num_sentences, 1, 1]
    return flattened_emb[indices + offset]


def print_range(text, i, j):
    while i <= j:
        print(f"{text[i]} ", end='')
        i = i + 1


def printexit(x):
    print(x)
    exit()
