import numpy as np
import collections
import torch
import util


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
    sentences_indices = torch.arange(num_sentences).unsqueeze(1).repeat([1, max_num_spans]) # [num_sentences, max_num_spans]
    sparse_indices = torch.cat([
        sentences_indices.unsqueeze(2),
        span_starts.unsqueeze(2),
        span_ends.unsqueeze(3)
    ])
    sparse_indices = tf.concat([
        tf.expand_dims(sentence_indices, 2),
        tf.expand_dims(span_starts, 2),
        tf.expand_dims(span_ends, 2)], axis=2)  # [num_sentences, max_num_spans, 3]
    if span_parents is not None:
        sparse_indices = tf.concat([
            sparse_indices, tf.expand_dims(span_parents, 2)], axis=2)  # [num_sentenes, max_num_spans, 4]

    rank = 3 if (span_parents is None) else 4
    # (sent_id, span_start, span_end) -> span_label
    dense_labels = tf.sparse_to_dense(
        sparse_indices=tf.reshape(sparse_indices, [num_sentences * max_num_spans, rank]),
        output_shape=[num_sentences] + [max_sentence_length] * (rank - 1),
        sparse_values=tf.reshape(span_labels, [-1]),
        default_value=0,
        validate_indices=False)  # [num_sentences, max_sent_len, max_sent_len]
    return dense_labels



