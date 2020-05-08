import numpy as np
import collections
import torch


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
        print(char_vocab_path)
        print(f.readlines())
        vocab.extend(c for c in f.readlines())

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


def pad_batch_tensors(tensors):
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

    return np.concatenate(padded_tensors, axis=0)
