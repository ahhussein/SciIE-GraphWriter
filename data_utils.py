import numpy as np
import collections
import torch
import util
import operator



class EmbeddingDictionary(object):
    def __init__(self, info, normalize=True, maybe_cache=None):
        self._size = info["size"]
        self._lowercase = info["lowercase"]
        self._normalize = normalize
        self._path = info["path"]

        if maybe_cache is not None and maybe_cache._path == self._path:
            assert self._size == maybe_cache._size
            assert self._lowercase == maybe_cache.lowercase
            self._embeddings = maybe_cache._embeddings
        else:
            self._embeddings = self.load_embedding_dict(self._path, info["format"])

    @property
    def size(self):
        return self._size

    def load_embedding_dict(self, path, file_format):
        """
        Load a dict of word: <embedding> {WORD: [...]}
        """

        print("Loading word embeddings from {}...".format(path))

        default_embedding = torch.zeros(self.size)
        embedding_dict = collections.defaultdict(lambda: default_embedding)

        if len(path) > 0:
            is_vec = (file_format == "vec")
            vocab_size = None

            with open(path, "r") as f:
                i = 0
                for line in f:
                    splits = line.split()
                    if i == 0 and is_vec:
                        vocab_size = int(splits[0])
                        assert int(splits[1]) == self.size
                    else:
                        assert len(splits) == self.size + 1
                        word = splits[0]
                        embedding = torch.tensor([float(s) for s in splits[1:]])
                        embedding_dict[word] = embedding
                    i += 1
                f.close()

            if vocab_size is not None:
                assert vocab_size == len(embedding_dict)

            print("Done loading word embeddings.")

        return embedding_dict

    def __getitem__(self, key):
        if self._lowercase:
            key = key.lower()
        embedding = self._embeddings[key]
        if self._normalize:
            embedding = self.normalize(embedding)
        return embedding

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        else:
            return v


def load_lm_embeddings_for_sentence(
        lm_file,
        doc_key,
        sent_key,
        transpose):
    """
    Load LM embeddings for a given sentence.
    """
    # TODO conver to torch
    # TODO preprocessed embeddings that needs to be adjusted with different datasets
    file_key = doc_key.replace("/", ":")

    group = lm_file[file_key]
    if sent_key is not None:
        sentence = group[sent_key][...]
    else:
        sentence = group[...]

    if transpose:
        return sentence.transpose(1, 2, 0)
    else:
        # words-in-sent x lm-size (1024) x lm-layers (3)
        return sentence


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with open(char_vocab_path) as f:
        vocab.extend(c.rstrip("\n") for c in f.readlines())
    f.close()
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})

    return char_dict


def tensorize_labeled_spans(tuples, label_dict=None):
    """
    Represent start, end, label data into three tensors
    convert labels to indices from label_dict if provided
    """
    if len(tuples) > 0:
        # Unfolding tuples of ner and coref
        starts, ends, labels = zip(*tuples)
    else:
        starts, ends, labels = [], [], []

    if label_dict:
        return torch.tensor(starts), torch.tensor(ends), torch.tensor([label_dict.get(c, 0) for c in labels])

    return torch.tensor(starts), torch.tensor(ends), torch.tensor(labels)


def tensorize_entity_relations(tuples, label_dict, filter_reverse):
    """
    Represent e1_start, e1_end, e2_start, e2_end, and label into 5 tensors
    convert labels to indices from label_dict if provided
    """
    # Removing V-V self-loop.
    filtered_tuples = []
    for t in tuples:
        if filter_reverse and "REVERSE" in t[-1]:
            filtered_tuples.append(t[:-1] + [t[-1].split("_REVERSE")[0], ])
        else:
            filtered_tuples.append(t)

    if len(filtered_tuples) > 0:
        s1, e1, s2, e2, labels = zip(*filtered_tuples)
    else:
        s1, e1, s2, e2, labels = [], [], [], [], []

    return (torch.tensor(s1), torch.tensor(e1), torch.tensor(s2), torch.tensor(e2),
            torch.tensor([label_dict.get(c, 0) for c in labels]))


def pad_batch_tensors(tensors, convert_tensor=True):
    """
    Args:
      tensors: List of tensors: numpy array of length B.

    Returns:
      Numpy array of (B, ?)
    """
    tensors = [np.expand_dims(tensor, 0) for tensor in tensors]

    shapes = [t.shape for t in tensors]

    # Take max shape along each dimension.
    max_shape = [max(item) for item in zip(*shapes)]

    pad_shapes = [tuple([tuple((0, a_i - b_i)) for a_i, b_i in zip(max_shape, shape)]) for shape in shapes]

    padded_tensors = [np.pad(tensor, pad_shape, "constant") for tensor, pad_shape in zip(tensors, pad_shapes)]

    if convert_tensor:
        return torch.from_numpy(np.concatenate(padded_tensors, axis=0))
    return np.concatenate(padded_tensors, axis=0)


def get_span_candidates(text_len, max_sentence_len, max_mention_width):
    """
    Params:
        text_len: Tensor of [num_sentences,] and it holds sentence lengths in terms of words
        max_sentence_len: the maximum sentence length
        max_mention_width: the maximum allowed mention width
    Returns:
        candidate_starts: tensor of all possible candidate start [num_sentences, max_mention_width * max_sentence_length]
        candidate_ends: tensor of all possible span ends [num_sentences, max_mention_width * max_sentence_length]
        candidate_mask:  Mask to respect the individual sentence lengths
    """

    num_sentences = text_len.shape[0]

    # [num_sentences, max_mention_width, max_sentence_length]
    candidate_starts = torch.arange(0, max_sentence_len).unsqueeze(0).unsqueeze(1).repeat(
        num_sentences, max_mention_width, 1
    )

    # [1, max_mention_width, 1]
    candidate_widths = torch.arange(0, max_mention_width).unsqueeze(0).unsqueeze(2)

    # [num_sentences, max_mention_width, max_sentence_length]
    candidate_ends = candidate_starts + candidate_widths

    # Reshaping both to: [num_sentences, max_mention_width * max_sentence_length]
    candidate_starts = candidate_starts.view(num_sentences, max_mention_width * max_sentence_len)
    candidate_ends = candidate_ends.view(num_sentences, max_mention_width * max_sentence_len)

    # [num_sentences, max_mention_width * max_sentence_length]
    mask_base = text_len.unsqueeze(1).repeat(1, max_mention_width * max_sentence_len)

    candidate_mask = candidate_ends < mask_base

    # Mask to avoid indexing error.
    # mask start to zero out the corresponding start index
    candidate_starts = torch.mul(candidate_starts, candidate_mask)
    candidate_ends = torch.mul(candidate_ends, candidate_mask)

    return candidate_starts, candidate_ends, candidate_mask


def get_span_task_labels(arg_starts, arg_ends, batch, max_sentence_length):
    """Get dense labels for NER/Constituents (unary span prediction tasks).
    """
    num_sentences = arg_starts.shape[0]
    max_num_args = arg_starts.shape[1]
    # [num_sentences, max_num_args]
    sentence_indices = torch.arange(num_sentences).unsqueeze(1).repeat([1, max_num_args])

    # [num_sentences, max_num_args, 3]
    pred_indices = torch.cat([
        sentence_indices.unsqueeze(2),
        arg_starts.unsqueeze(2),
        arg_ends.unsqueeze(2)], 2
    )

    dense_ner_labels = get_dense_span_labels(
        batch.ner_starts, batch.ner_ends, batch.ner_labels, batch.ner_len,
        max_sentence_length)  # [num_sentences, max_sent_len, max_sent_len]

    dense_coref_labels = get_dense_span_labels(
        batch.coref_starts, batch.coref_ends, batch.coref_cluster_ids, batch.coref_len,
        max_sentence_length)  # [num_sentences, max_sent_len, max_sent_len]

    ner_labels = gather_nd(dense_ner_labels, pred_indices)  # [num_sentences, max_num_args]
    coref_cluster_ids = gather_nd(dense_coref_labels, pred_indices)  # [num_sentences, max_num_args]
    return ner_labels, coref_cluster_ids


def get_dense_span_labels(span_starts, span_ends, span_labels, num_spans, max_sentence_length, span_parents=None):
    """Utility function to get dense span or span-head labels.
    Args:
      span_starts: [num_sentences, max_num_spans]
      span_ends: [num_sentences, max_num_spans]
      span_labels: [num_sentences, max_num_spans]
      num_spans: [num_sentences,]
      max_sentence_length:
      span_parents: [num_sentences, max_num_spans]. Predicates in SRL.
    """
    num_sentences = span_starts.shape[0]
    max_num_spans = span_starts.shape[1]

    # For padded spans, we have starts = 1, and ends = 0, so they don't collide with any existing spans.
    span_starts += (1 - util.sequence_mask(num_spans, dtype=torch.int32))  # [num_sentences, max_num_spans]
    sentences_indices = torch.arange(num_sentences).unsqueeze(1).repeat(
        [1, max_num_spans])  # [num_sentences, max_num_spans]

    sparse_indices = torch.cat([
        sentences_indices.unsqueeze(2),
        span_starts.unsqueeze(2),
        span_ends.unsqueeze(2)
    ], 2)  # [num_sentences, max_num_spans, 3]

    if span_parents is not None:
        # [num_sentenes, max_num_spans, 4]
        sparse_indices = torch.cat([sparse_indices, span_parents.unsqueeze(2)], 2)

    rank = 3 if (span_parents is None) else 4

    sparse_indices = sparse_indices.view([num_sentences * max_num_spans, rank])
    sparse_values = span_labels.view(-1)
    dense_labels = torch.zeros([num_sentences] + [max_sentence_length] * (rank - 1), dtype=torch.int32)

    for i in range(len(sparse_indices)):
        dense_labels[tuple(sparse_indices[i])] = sparse_values[i]

    # (sent_id, span_start, span_end) -> span_label
    return dense_labels


def gather_nd(params, indices):
    '''
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices

    returns: tensor shaped [m_1, m_2, m_3, m_4]

    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices

    returns: tensor shaped [m_1, ..., m_1]
    '''

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)


def get_relation_labels(entity_starts, entity_ends, num_entities, max_sentence_length,
                        gold_e1_starts, gold_e1_ends, gold_e2_starts, gold_e2_ends,
                        gold_labels, num_gold_rels):
    num_sentences, max_num_ents = entity_starts.shape
    rel_labels = torch.zeros([num_sentences, max_num_ents + 1, max_num_ents + 1], dtype=torch.int64)
    entity_ids = torch.zeros([num_sentences, max_sentence_length, max_sentence_length], dtype=torch.int64)

    for i in range(num_sentences):
        for j in range(num_entities[i]):
            entity_ids[i, entity_starts[i, j], entity_ends[i, j]] = j + 1
        for j in range(num_gold_rels[i]):
            rel_labels[i, entity_ids[i, gold_e1_starts[i, j], gold_e1_ends[i, j]],
                       entity_ids[i, gold_e2_starts[i, j], gold_e2_ends[i, j]]] = gold_labels[i, j]
    return rel_labels[:, 1:, 1:]  # Remove "dummy" entities.


def get_coref_softmax_loss(antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + torch.log(antecedent_labels.type(torch.float32))  # [k, max_ant + 1]
    marginalized_gold_scores = torch.logsumexp(gold_scores, [1])  # [k]
    log_norm = torch.logsumexp(antecedent_scores, [1])  # [k]
    return log_norm - marginalized_gold_scores  # [k]


def split_example_for_eval(example):
  """Split document-based samples into sentence-based samples for evaluation.

  Args:
    example:
  Returns:
    Tuple of (sentence, list of SRL relations)
  """
  sentences = example["sentences"]

  if "srl" not in example:
    example["srl"] = [[] for s in sentences]

  if "relations" not in example:
    example["relations"] = [[] for s in sentences]

  word_offset = 0
  samples = []

  for i, sentence in enumerate(sentences):
    srl_rels = {}
    ner_spans = []
    relations = []

    for r in example["srl"][i]:
      pred_id = r[0] - word_offset
      if pred_id not in srl_rels:
        srl_rels[pred_id] = []
      srl_rels[pred_id].append((r[1] - word_offset, r[2] - word_offset, r[3]))

    for r in example["ner"][i]:
      ner_spans.append((r[0] - word_offset, r[1] - word_offset, r[2]))

    for r in example["relations"][i]:
      relations.append((r[0] - word_offset, r[1] - word_offset, r[2] - word_offset,
                        r[3] - word_offset, r[4]))
    samples.append((sentence, srl_rels, ner_spans, relations))
    word_offset += len(sentence)

  return samples

class RetrievalEvaluator(object):
  def __init__(self):
    self._num_correct = 0
    self._num_gold = 0
    self._num_predicted = 0

  def update(self, gold_set, predicted_set):
    self._num_correct += len(gold_set & predicted_set)
    self._num_gold += len(gold_set)
    self._num_predicted += len(predicted_set)

  def recall(self):
    return maybe_divide(self._num_correct, self._num_gold)

  def precision(self):
    return maybe_divide(self._num_correct, self._num_predicted)

  def metrics(self):
    recall = self.recall()
    precision = self.precision()
    f1 = maybe_divide(2 * recall * precision, precision + recall)
    return recall, precision, f1

def maybe_divide(x, y):
  return 0 if y == 0 else x / float(y)

def evaluate_retrieval(span_starts, span_ends, span_scores, pred_starts, pred_ends, gold_spans,
                       text_length, evaluators, debugging=False):
  """
  Evaluation for unlabeled retrieval.

  Args:
    gold_spans: Set of tuples of (start, end).
  """
  if len(span_starts) > 0:
    sorted_starts, sorted_ends, sorted_scores = zip(*sorted(
        list(zip(span_starts, span_ends, span_scores)),
        key=operator.itemgetter(2), reverse=True))
  else:
    sorted_starts = []
    sorted_ends = []
  for k, evaluator in evaluators.items():
    if k == -3:
      predicted_spans = set(zip(span_starts, span_ends)) & gold_spans
    else:
      if k == -2:
        predicted_starts = pred_starts
        predicted_ends = pred_ends
        if debugging:
          print("Predicted", list(zip(sorted_starts, sorted_ends, sorted_scores))[:len(gold_spans)])
          print("Gold", gold_spans)
     # FIXME: scalar index error
      elif k == 0:
        is_predicted = span_scores > 0
        predicted_starts = span_starts[is_predicted]
        predicted_ends = span_ends[is_predicted]
      else:
        if k == -1:
          num_predictions = len(gold_spans)
        else:
          num_predictions = int((k * text_length) / 100)


        predicted_starts = sorted_starts[:num_predictions]
        predicted_ends = sorted_ends[:num_predictions]
      predicted_spans = set(zip(predicted_starts, predicted_ends))
    evaluator.update(gold_set=gold_spans, predicted_set=predicted_spans)


def compute_relation_f1(gold_rels, predictions):
    assert len(gold_rels) == len(predictions)

    total_gold = 0
    total_predicted = 0
    total_matched = 0
    total_unlabeled_matched = 0
    # Compute unofficial F1 of entity relations.
    doc_id = 0  # Actually sentence id.
    gold_tuples = []  # For official eval.
    predicted_tuples = []

    for gold, prediction in list(zip(gold_rels, predictions)):
        total_gold += len(gold)
        total_predicted += len(prediction)
        for g in gold:
            gold_tuples.append([["d{}_{}_{}".format(doc_id, g[0], g[1]),
                                 "d{}_{}_{}".format(doc_id, g[2], g[3])], g[4]])
            for p in prediction:
                if g[0] == p[0] and g[1] == p[1] and g[2] == p[2] and g[3] == p[3]:
                    total_unlabeled_matched += 1
                    if g[4] == p[4]:
                        total_matched += 1
                    break

        for p in prediction:
            predicted_tuples.append([["d{}_{}_{}".format(doc_id, p[0], p[1]),
                                      "d{}_{}_{}".format(doc_id, p[2], p[3])], p[4]])
        doc_id += 1

    precision, recall, f1 = util._print_f1(total_gold, total_predicted, total_matched, "Relations (unofficial)")
    util._print_f1(total_gold, total_predicted, total_unlabeled_matched,
                                          "Unlabeled (unofficial)")
    util.span_metric(gold_tuples, predicted_tuples)
    return precision, recall, f1


def compute_span_f1(gold_data, predictions, task_name):
  assert len(gold_data) == len(predictions)
  total_gold = 0
  total_predicted = 0
  total_matched = 0
  total_unlabeled_matched = 0
  label_confusions = collections.Counter()  # Counter of (gold, pred) label pairs.

  for i in range(len(gold_data)):
    gold = gold_data[i]
    pred = predictions[i]
    total_gold += len(gold)
    total_predicted += len(pred)

    for a0 in gold:
      for a1 in pred:
        if a0[0] == a1[0] and a0[1] == a1[1]:
          total_unlabeled_matched += 1
          label_confusions.update([(a0[2], a1[2]),])
          if a0[2] == a1[2]:
            total_matched += 1

  prec, recall, f1 = util._print_f1(total_gold, total_predicted, total_matched, task_name)
  ul_prec, ul_recall, ul_f1 = util._print_f1(total_gold, total_predicted, total_unlabeled_matched, "Unlabeled " + task_name)
  return prec, recall, f1, ul_prec, ul_recall, ul_f1, label_confusions

