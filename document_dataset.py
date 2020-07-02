from torchtext import data
import numpy as np
import json
import util
import data_utils
import random
import h5py

import torch

# Names for the "given" tensors.
_input_names = [
    "tokens", "context_word_emb", "head_word_emb", "lm_emb", "char_idx", "text_len", "doc_id", "is_training"]

_label_names = [
    "ner_starts", "ner_ends", "ner_labels", "ner_len",
    "coref_starts", "coref_ends", "coref_cluster_ids", "coref_len",
    "rel_e1_starts", "rel_e1_ends", "rel_e2_starts", "rel_e2_ends", "rel_labels", "rel_len"
]

_predict_names = [
    "candidate_starts", "candidate_ends", "candidate_arg_scores", "candidate_pred_scores", "ner_scores", "arg_scores", "pred_scores",
    "candidate_mention_starts", "candidate_mention_ends", "candidate_mention_scores", "mention_starts",
    "mention_ends", "antecedents", "antecedent_scores",
    "srl_head_scores", "coref_head_scores", "ner_head_scores", "entity_gate", "antecedent_attn",
    # Relation stuff.
    "candidate_entity_scores", "entity_starts", "entity_ends", "entitiy_scores", "num_entities",
    "rel_labels", "rel_scores",
]

class DocumentDataset(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.lm_file = h5py.File(self.config["lm_path"], "r")
        self.lm_layers = self.config["lm_layers"]
        self.lm_size = self.config["lm_size"]
        self.ner_labels = {l: i for i, l in enumerate([""] + config["ner_labels"])}
        self.ner_labels_inv = [""] + config["ner_labels"]

        self.input_names = _input_names
        self.label_names = _label_names
        self.predict_names = _predict_names

        # Fields declaration
        self.fields = [
            ("tokens", data.RawField()),
            ("context_word_emb", data.RawField()),
            ("head_word_emb", data.RawField()),
            ("lm_emb", data.RawField()),
            ("char_idx", data.RawField()),
            ("text_len", data.RawField()),
            ("doc_id", data.RawField()),
            ("doc_key", data.RawField()),
            ("is_training", data.RawField()),
            ("ner_starts", data.RawField()),
            ("ner_ends", data.RawField()),
            ("ner_labels", data.RawField()),
            ("coref_starts", data.RawField()),
            ("coref_ends", data.RawField()),
            ("coref_cluster_ids", data.RawField()),
            ("rel_e1_starts", data.RawField()),
            ("rel_e1_ends", data.RawField()),
            ("rel_e2_starts", data.RawField()),
            ("rel_e2_ends", data.RawField()),
            ("rel_labels", data.RawField()),
            ("ner_len", data.RawField()),
            ("coref_len", data.RawField()),
            ("rel_len", data.RawField()),
            ("title", data.RawField())
        ]

        self.rel_labels_inv = [""] + config["relation_labels"]
        if config["filter_reverse_relations"]:
            self.rel_labels_inv = [r for r in self.rel_labels_inv if "REVERSE" not in r]

        self.rel_labels = {l: i for i, l in enumerate(self.rel_labels_inv)}

        # TODO Understand the difference between the two glove files
        self.context_embeddings = data_utils.EmbeddingDictionary(config["context_embeddings"])
        #
        self.head_embeddings = data_utils.EmbeddingDictionary(
            config["head_embeddings"],
            maybe_cache=self.context_embeddings
        )

        self.char_embedding_size = config["char_embedding_size"]
        self.char_dict = data_utils.load_char_dict(config["char_vocab_path"])
        self.dict_size = len(self.char_dict)
        self.examples = []
        self._load_data()

        super(DocumentDataset, self).__init__(self.examples, self.fields)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def sort_key(ex):
        return len(ex.text_len)

    def _load_data(self):
        pass

    def _read_documents(self, train_examples):
        # List of documents, each holds a list of sentences.
        doc_examples = []

        cluster_id_offset = 0
        num_sentences = 0
        num_mentions = 0

        for doc_id, document in enumerate(train_examples, 1):
            doc_examples.append([])

            # Read sentences in a document
            for example in self._split_document_example(document):
                # append further attributes to sent in a document
                example["doc_id"] = doc_id
                example["cluster_id_offset"] = cluster_id_offset
                doc_examples[-1].append(example)
                num_mentions += len(example["coref"])

            cluster_id_offset += len(document["clusters"])
            num_sentences += len(doc_examples[-1])

        print("Load {} training documents with {} sentences, {} clusters, and {} mentions.".format(
            doc_id, num_sentences, cluster_id_offset, num_mentions))

        for doc_sentences in doc_examples:
            # TODO is_training should be class-level dynamic
            [self.tensorize_example(doc_sentence, is_training=True) for doc_sentence in doc_sentences]

    def _split_document_example(self, example):
        """
        Split document-based samples into sentence-based samples.
        """
        clusters = example["clusters"]

        # mentions sorted
        gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))

        cluster_ids = {}
        for cluster_id, cluster in enumerate(clusters, 1):
            for mention in cluster:
                cluster_ids[tuple(mention)] = cluster_id

        sentences = example["sentences"]
        split_examples = []
        word_offset = 0

        for i, sentence in enumerate(sentences):
            text_len = len(sentence)
            coref_mentions = []

            for start, end in gold_mentions:
                # Mention started with the start of the sentences and ended before the end of the sentence
                if word_offset <= start and end < word_offset + text_len:
                    coref_mentions.append([start, end, cluster_ids[(start, end)]])

            sent_example = {
                "sentence": sentence,
                "sent_id": i,
                "ner": example["ner"][i] if "ner" in example else [],
                "relations": example["relations"][i] if "relations" in example else [],
                "coref": coref_mentions,
                "word_offset": word_offset, # Sentence offset in the document
                "doc_key": example["doc_key"],
                "title": 'Dummy title', # TODO title
                "sent_offset": example["sent_offset"]  # Sentence offset for the same doc ID.
            }
            word_offset += text_len
            split_examples.append(sent_example)


        return split_examples

    def populate_sentence_offset(self, examples):
        """
        Compute sentence offset (that share the same doc key), because of LM embedding formatting difference.
        """
        prev_doc_key = "XXX"
        sent_offset = 0
        for example in examples:
            doc_key = example["doc_key"][:example["doc_key"].rfind("_")]
            if doc_key != prev_doc_key:
                prev_doc_key = doc_key
                sent_offset = 0
            example["sent_offset"] = sent_offset

            # number of sentences
            sent_offset += len(example["sentences"])

    def tensorize_example(self, example, is_training):
        """
        Tensorize examples and cache embeddings.
        """
        sentence = example["sentence"]
        doc_key = example["doc_key"]
        sent_id = example["sent_id"]
        word_offset = example["word_offset"]
        text_len = len(sentence)

        lm_doc_key = doc_key + "_" + str(sent_id)
        transpose = False

        # Load cached LM.
        lm_emb = data_utils.load_lm_embeddings_for_sentence(
            self.lm_file,
            lm_doc_key,
            None,
            transpose
        )

        max_word_length = max(max(len(w) for w in sentence), max(self.config["filter_widths"]))

        # Prepare context word embedding for a sentence
        context_word_emb = torch.zeros([text_len, self.context_embeddings.size])
        head_word_emb = torch.zeros([text_len, self.head_embeddings.size])
        char_index = torch.zeros([text_len, max_word_length], dtype=torch.long)

        for j, word in enumerate(sentence):
            context_word_emb[j] = self.context_embeddings[word]
            head_word_emb[j] = self.head_embeddings[word]

            # Rest is padded with zeros
            char_index[j, :len(word)] = torch.tensor([self.char_dict[c] for c in word])

        ner_starts, ner_ends, ner_labels = data_utils.tensorize_labeled_spans(
            example["ner"],
            self.ner_labels
        )

        coref_starts, coref_ends, coref_cluster_ids = data_utils.tensorize_labeled_spans(
            example["coref"]
        )

        rel_e1_starts, rel_e1_ends, rel_e2_starts, rel_e2_ends, rel_labels = (
            data_utils.tensorize_entity_relations(
                example["relations"],
                self.rel_labels,
                filter_reverse=self.config["filter_reverse_relations"]
            )
        )

        example = data.Example.fromlist([
                # Inputs
                sentence, # words in sentence (words-in-sent)
                context_word_emb, # words-in-sent x emb-size
                head_word_emb, # words-in-sent x emb-size
                lm_emb, # Lm embeddings for a sentence (words-in-sent x lm-size x lm-layers)
                char_index, # words-in-sent x max-word-length
                text_len,
                example["doc_id"],
                example["doc_key"],
                is_training,

                # Labels.
                ner_starts - word_offset,
                ner_ends - word_offset,
                ner_labels,
                coref_starts - word_offset,
                coref_ends - word_offset,
                coref_cluster_ids + example["cluster_id_offset"],
                rel_e1_starts - word_offset,
                rel_e1_ends - word_offset,
                rel_e2_starts - word_offset,
                rel_e2_ends - word_offset,
                rel_labels, # e.g. "conjunction"
                len(ner_starts), # entities
                len(coref_starts), # mentions
                len(rel_e1_starts), # relations,
                example['title']
            ], self.fields)


        self.examples.append(example)

        return example


    def fix_batch(self, batch):
        for field in self.fields:
            convert_tensor = True
            if field in ['tokens', 'doc_key', 'title']:
                convert_tensor = False
            setattr(batch, field, data_utils.pad_batch_tensors(getattr(batch, field), convert_tensor))

        batch.ner_starts = batch.ner_starts.type(torch.int64)
        batch.ner_ends = batch.ner_ends.type(torch.int64)
        batch.coref_starts = batch.coref_starts.type(torch.int64)
        batch.coref_ends = batch.coref_ends.type(torch.int64)
        batch.rel_e1_starts = batch.rel_e1_starts.type(torch.int64)
        batch.rel_e1_ends = batch.rel_e1_ends.type(torch.int64)
        batch.rel_e2_starts = batch.rel_e2_starts.type(torch.int64)
        batch.rel_e2_ends = batch.rel_e2_ends.type(torch.int64)

        return batch

class TrainDataset(DocumentDataset):
    def _load_data(self):
        with open(self.config["train_path"], "r", encoding='utf8') as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        self.populate_sentence_offset(train_examples)

        self._read_documents(train_examples)

class EvalDataset(DocumentDataset):
    def __init__(self, **kwargs):
        self.eval_data = None
        self.coref_eval_data = None
        super(EvalDataset, self).__init__(**kwargs)

    def _load_data(self):
        eval_data = {}
        coref_eval_data = {}

        with open(self.config["eval_path"]) as f:
            eval_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        self.populate_sentence_offset(eval_examples)

        for doc_id, document in enumerate(eval_examples,1):
            num_mentions_in_doc = 0

            for example in self._split_document_example(document):
                # Because each batch=1 document at test time, we do not need to offset cluster ids.
                example["cluster_id_offset"] = 0
                example["doc_id"] = doc_id
                self.tensorize_example(example, is_training=False)
                num_mentions_in_doc += len(example["coref"])

            assert num_mentions_in_doc == len(util.flatten(document["clusters"]))

            eval_data[doc_id] = data_utils.split_example_for_eval(document)
            coref_eval_data[doc_id] = document

        print("Loaded {} eval examples.".format(len(eval_data)))
        self.eval_data = eval_data
        self.coref_eval_data = coref_eval_data





