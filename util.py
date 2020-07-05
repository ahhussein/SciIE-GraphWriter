import pyhocon
import os
import errno
import torch
import span_prune_cpp
import sys
import codecs
import subprocess
import math
from sklearn.metrics import classification_report, precision_recall_fscore_support

from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response

from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import Iterator

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
    num_sentences = candidate_starts.shape[0]

    # [num_sentences]
    topk = torch.max(
        torch.floor(text_len.type(torch.float32) * topk_ratio).type(torch.int32),
        torch.ones(num_sentences, dtype=torch.int32)
    )

    # [num_sentences, pruned_num_candidates = max(topk)]
    min = -sys.maxsize - 1
    candidate_scores[candidate_scores == float('-inf')] = min

    # TODO: go through the pruning algorithm
    predicted_indices = span_prune_cpp.extract_spans(
        candidate_scores,
        candidate_starts,
        candidate_ends,
        topk,
        max_sentence_length,
        sort_spans,
        enforce_non_crossing
    )


    # TODO why repeated indices?
    predicted_indices = predicted_indices.type(torch.int64)
    # get corresponding span starts and ends
    predicted_starts = batch_gather(candidate_starts, predicted_indices)  # [num_sentences, pruned_num_candidates]
    predicted_ends = batch_gather(candidate_ends, predicted_indices)  # [num_sentences, pruned_num_candidates]

    predicted_scores = batch_gather(candidate_scores, predicted_indices)  # [num_sentences, pruned_num_candidates]

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

def bucket_distance(distances):
  """
  Places the given values (designed for distances) into 10 semi-logscale buckets:
  [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
  """
  logspace_idx = torch.floor((torch.log(distances.type(torch.float32))/math.log(2))).type(torch.int32) + 3
  use_identity = (distances <= 4).type(torch.int32)
  combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
  return torch.min(combined_idx, torch.tensor(9))


def response_chunks(response, chunk_size=CONTENT_CHUNK_SIZE):
    # type: (Response, int) -> Iterator[bytes]
    """Given a requests Response, provide the data chunks.
    """
    try:
        # Special case for urllib3.
        for chunk in response.raw.stream(
            chunk_size,
            # We use decode_content=False here because we don't
            # want urllib3 to mess with the raw bytes we get
            # from the server. If we decompress inside of
            # urllib3 then we cannot verify the checksum
            # because the checksum will be of the compressed
            # file. This breakage will only occur if the
            # server adds a Content-Encoding header, which
            # depends on how the server was configured:
            # - Some servers will notice that the file isn't a
            #   compressible file and will leave the file alone
            #   and with an empty Content-Encoding
            # - Some servers will notice that the file is
            #   already compressed and will leave the file
            #   alone and will add a Content-Encoding: gzip
            #   header
            # - Some servers won't notice anything at all and
            #   will take a file that's already been compressed
            #   and compress it again and set the
            #   Content-Encoding: gzip header
            #
            # By setting this not to decode automatically we
            # hope to eliminate problems with the second case.
            decode_content=False,
        ):
            yield chunk
    except AttributeError:
        # Standard file-like object.
        while True:
            chunk = response.raw.read(chunk_size)
            if not chunk:
                break
            yield chunk

def _print_f1(total_gold, total_predicted, total_matched, message=""):
  precision = 100.0 * total_matched / total_predicted if total_predicted > 0 else 0
  recall = 100.0 * total_matched / total_gold if total_gold > 0 else 0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
  print ("{}: Precision: {}, Recall: {}, F1: {}".format(message, precision, recall, f1))
  return precision, recall, f1


def span_metric(grelations, prelations):
    g_spans = []
    p_spans = []
    res_gold = []
    res_pred = []
    for rel in grelations:
        if 'REVERSE' in rel[1]:
            span = rel[0][1] + '_' + rel[0][0]
            g_spans.append(span)
        else:
            g_spans.append('_'.join(rel[0]))
        res_gold.append(rel[1])

    for rel in prelations:
        if 'REVERSE' in rel[1]:
            span = rel[0][1] + '_' + rel[0][0]
            p_spans.append(span)
        else:
            p_spans.append('_'.join(rel[0]))
        res_pred.append(rel[1])

    spans_all = set(p_spans + g_spans)
    res_all_gold = []
    res_all_pred = []
    targets = set()
    for i, r in enumerate(spans_all):
        if r in g_spans:
            target = res_gold[g_spans.index(r)].split("_")[0]
            res_all_gold.append(target)
            targets.add(target)
        else:
            res_all_gold.append('None')
        if r in p_spans:
            target = res_pred[p_spans.index(r)].split("_")[0]
            res_all_pred.append(target)
            targets.add(target)
        else:
            res_all_pred.append('None')
    targets = list(targets)
    prec, recall, f1, support = precision_recall_fscore_support(res_all_gold, res_all_pred, labels=targets,
                                                                average=None)
    metrics = {}
    metrics = {}
    for k, target in enumerate(targets):
        metrics[target] = {
            'precision': prec[k],
            'recall': recall[k],
            'f1-score': f1[k],
            'support': support[k]
        }
    prec, recall, f1, s = precision_recall_fscore_support(res_all_gold, res_all_pred, labels=targets, average='micro')

    metrics['overall'] = {
        'precision': prec,
        'recall': recall,
        'f1-score': f1,
        'support': sum(support)
    }
    # print_report(metrics, targets)
    return prec, recall, f1


def print_report(metrics, targets, digits=2):
    def _get_line(results, target, columns):
        line = [target]
        for column in columns[:-1]:
            line.append("{0:0.{1}f}".format(results[column], digits))
        line.append("%s" % results[columns[-1]])
        return line

    columns = ['precision', 'recall', 'f1-score', 'support']

    fmt = '%11s' + '%9s' * 4 + '\n'
    report = [fmt % tuple([''] + columns)]
    report.append('\n')
    for target in targets:
        results = metrics[target]
        line = _get_line(results, target, columns)
        report.append(fmt % tuple(line))
    report.append('\n')

    # overall
    line = _get_line(metrics['overall'], 'avg / total', columns)
    report.append(fmt % tuple(line))
    report.append('\n')

    print(''.join(report))


def print_to_iob2(sentences, gold_ner, pred_ner, gold_file_path):
  """Print to IOB2 format for NER eval.
  """
  # Write NER prediction to IOB format.
  temp_file_path = "/tmp/ner_pred_%d.tmp" % os.getpid()
  # Read IOB tags from preprocessed gold path.
  gold_info = [[]]

  if gold_file_path:
    fgold = codecs.open(gold_file_path, "r", "utf-8")
    for line in fgold:
      line = line.strip()
      if not line:
        gold_info.append([])
      else:
        gold_info[-1].append(line.split())
  else:
    fgold = None

  fout = codecs.open(temp_file_path, "w", "utf-8")

  for sent_id, words in enumerate(sentences):
    pred_tags = ["O" for _ in words]
    for start, end, label in pred_ner[sent_id]:
      pred_tags[start] = "B-" + label
      for k in range(start + 1, end + 1):
        pred_tags[k] = "I-" + label

    if not fgold:
      gold_tags = ["O" for _ in words]
      for start, end, label in gold_ner[sent_id]:
        gold_tags[start] = "B-" + label
        for k in range(start + 1, end + 1):
          gold_tags[k] = "I-" + label
    else:
      assert len(gold_info[sent_id]) == len(words)
      gold_tags = [t[1] for t in gold_info[sent_id]]

    for w, gt, pt in list(zip(words, gold_tags, pred_tags)):
      fout.write(w + " " + gt + " " + pt + "\n")

    fout.write("\n")

  fout.close()
  child = subprocess.Popen('./ner/bin/conlleval < {}'.format(temp_file_path),
                           shell=True, stdout=subprocess.PIPE)
  eval_info = child.communicate()[0]
  print(eval_info)

def print_range(text, i, j):
    while i <= j:
        print(f"{text[i]} ", end='')
        i = i + 1


def printexit(x):
    print(x)
    exit()
