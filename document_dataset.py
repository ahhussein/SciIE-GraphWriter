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
    "tokens", "context_word_emb", "head_word_emb", "lm_emb", "char_idx", "text_len", "doc_id"]

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

class DocumentDataset():
    def __init__(self, config, is_eval = False):
        self.config = config
        self.lm_layers = self.config["lm_layers"]
        self.lm_size = self.config["lm_size"]
        self.lm_file = h5py.File(config["lm_path"], "r")
        self.ner_labels = {l: i for i, l in enumerate([""] + config["ner_labels"])}
        self.ner_labels_inv = [""] + config["ner_labels"]

        self.input_names = _input_names
        self.label_names = _label_names
        self.predict_names = _predict_names


        self.title = data.Field(sequential=True, batch_first=True,init_token="<start>", eos_token="<eos>",include_lengths=True)
        self.out = data.Field(sequential=True, batch_first=True, init_token="<start>", eos_token="<eos>", include_lengths=True)

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
            ("title", self.title),
            ("doc_len", data.RawField()),
            ('out', self.out)
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
        self.eval_examples = []

        self._load_data()

        self.dataset = data.Dataset(self.examples, self.fields)

        # If eval, don't load data. Just build the vocab
        if not is_eval:
            # Update lm_file to read the val data
            self.lm_file = h5py.File(config["lm_path_dev"], "r")
            self._load_eval_data()
            self.val_dataset = data.Dataset(self.eval_examples, self.fields)

        if is_eval:
            self.lm_file = h5py.File(config["lm_path_test"], "r")
            self._load_test_data()
            self.test_dataset = data.Dataset(self.eval_examples, self.fields)

        self._build_vocab()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def sort_key(ex):
        return len(ex.text_len)

    def _load_data(self):
        with open(self.config["train_path"], "r", encoding='utf8') as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        self.populate_sentence_offset(train_examples)

        self._read_documents(train_examples)

    def _load_eval_data(self):
        #coref_eval_data = {}

        # List of documents, each holds a list of sentences.
        doc_examples = []
        num_mentions = 0
        num_sentences = 0
        doc_count = 0
        cluster_id_offset = 0

        with open(self.config["eval_path"]) as f:
            eval_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        self.populate_sentence_offset(eval_examples)

        for doc_id, document in enumerate(eval_examples, 1):
            num_mentions_in_doc = 0
            doc_examples.append([])

            for example in self._split_document_example(document):
                example["cluster_id_offset"] = cluster_id_offset
                example["doc_id"] = doc_id
                doc_examples[-1].append(example)
                num_mentions_in_doc += len(example["coref"])
                num_mentions += len(example["coref"])

            assert num_mentions_in_doc == len(util.flatten(document["clusters"]))

            cluster_id_offset += len(document["clusters"])
            num_sentences += len(doc_examples[-1])
            doc_count += 1

            #coref_eval_data[doc_id] = document

        print("Loaded {} eval examples.".format(doc_count))

        for doc_sentences in doc_examples:
            doc_sentences_processed = []
            out_text = [word for sentence in doc_sentences for word in sentence['sentence']]

            # TODO title
            title = 'Dummy title'

            [doc_sentences_processed.append(self.tensorize_example(doc_sentence)) for doc_sentence in doc_sentences]

            example = data.Example.fromlist(
                (
                    np.stack(np.array(doc_sentences_processed), 1).tolist()
                ) + [
                    title, len(doc_sentences_processed), out_text
                ], self.fields)

            self.eval_examples.append(example)

        # TODO use later to sci-erc evaluation
        #self.coref_eval_data = coref_eval_data

    def _load_test_data(self):
        # List of documents, each holds a list of sentences.
        doc_examples = []
        num_mentions = 0
        num_sentences = 0
        doc_count = 0
        cluster_id_offset = 0

        with open(self.config["test_path"]) as f:
            eval_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        self.populate_sentence_offset(eval_examples)

        for doc_id, document in enumerate(eval_examples, 1):
            doc_examples.append([])

            for example in self._split_document_example(document):
                example["cluster_id_offset"] = cluster_id_offset
                example["doc_id"] = doc_id
                doc_examples[-1].append(example)
                num_mentions += len(example["coref"])

            cluster_id_offset += len(document["clusters"])
            num_sentences += len(doc_examples[-1])
            doc_count += 1

            #coref_eval_data[doc_id] = document

        print("Loaded {} eval examples.".format(doc_count))

        for doc_sentences in doc_examples:
            doc_sentences_processed = []
            out_text = [word for sentence in doc_sentences for word in sentence['sentence']]

            # TODO title
            title = 'Dummy title'

            [doc_sentences_processed.append(self.tensorize_example(doc_sentence)) for doc_sentence in doc_sentences]

            example = data.Example.fromlist(
                (
                    np.stack(np.array(doc_sentences_processed), 1).tolist()
                ) + [
                    title, len(doc_sentences_processed), out_text
                ], self.fields)

            self.eval_examples.append(example)

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
            doc_sentences_processed = []
            #out_text = [word for sentence in doc_sentences for word in sentence['sentence']]
            out_text = self.build_out_text_for_document(doc_sentences)
            # TODO title
            title = 'Dummy title'

            [doc_sentences_processed.append(self.tensorize_example(doc_sentence)) for doc_sentence in doc_sentences]

            example = data.Example.fromlist(
                (
                    np.stack(np.array(doc_sentences_processed),1).tolist()
                ) + [
                    title, len(doc_sentences_processed), out_text
                ]  , self.fields)

            self.examples.append(example)

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

    def tensorize_example(self, example):
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

        example = [
                # Inputs
                sentence, # words in sentence (words-in-sent)
                context_word_emb, # words-in-sent x emb-size
                head_word_emb, # words-in-sent x emb-size
                lm_emb, # Lm embeddings for a sentenfce (words-in-sent x lm-size x lm-layers)
                char_index, # words-in-sent x max-word-length
                text_len,
                example["doc_id"],
                example["doc_key"],

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
                len(rel_e1_starts), # relations
            ]

        return example

    def fix_batch(self, batch):
        for field in self.dataset.fields:
            convert_tensor = True
            if field in ['tokens', 'doc_key']:
                convert_tensor = False
            if field in ['doc_len', 'title', 'out']:
                continue

            setattr(batch, field, data_utils.pad_batch_tensors(getattr(batch, field), convert_tensor))

        batch.doc_len = torch.tensor(batch.doc_len)
        batch.ner_starts = batch.ner_starts.type(torch.int64)
        batch.ner_ends = batch.ner_ends.type(torch.int64)
        batch.coref_starts = batch.coref_starts.type(torch.int64)
        batch.coref_ends = batch.coref_ends.type(torch.int64)
        batch.rel_e1_starts = batch.rel_e1_starts.type(torch.int64)
        batch.rel_e1_ends = batch.rel_e1_ends.type(torch.int64)
        batch.rel_e2_starts = batch.rel_e2_starts.type(torch.int64)
        batch.rel_e2_ends = batch.rel_e2_ends.type(torch.int64)

        return batch

    def _build_vocab(self):
        self.title.build_vocab(self.dataset, min_freq=5)

        # Extend the output vocab to contain these tokens (They exist in the original vocab but with numbers)
        generics = ['<method>', '<material>', '<otherscientificterm>', '<metric>', '<task>']
        self.out.build_vocab(self.dataset, min_freq=5)
        self.out.vocab.itos.extend(generics)
        for x in generics:
            self.out.vocab.stoi[x] = self.out.vocab.itos.index(x)

        self.config.ntoks = len(self.out.vocab)

        # # Extend the outpt vocab to contain these tokens
        # generics = ['<method>', '<material>', '<otherscientificterm>', '<metric>', '<task>']
        # self.out.vocab.itos.extend(generics)
        # for x in generics:
        #     self.out.vocab.stoi[x] = self.out.vocab.itos.index(x)
        #
        # # Extend target to include all the entity types with numbers up to 40
        # specials = "method material otherscientificterm metric task".split(" ")
        # for x in specials:
        #     for y in range(40):
        #         s = "<" + x + "_" + str(y) + ">"
        #         self.out.vocab.itos.append(s)
        #         self.out.vocab.stoi[s] = len(self.out.vocab.itos)

    def reverse(self, x, ents):
        # TODO, ents
        #ents = ents[0]
        vocab = self.out.vocab
        s = ' '.join(
            [vocab.itos[y] if y < len(vocab.itos) else 0 for j, y in enumerate(x)])

        #s = ' '.join(
        #    [vocab.itos[y] if y < len(vocab.itos) else ents[y - len(vocab.itos)].upper() for j, y in enumerate(x)])
        # s = ' '.join([vocab.itos[y] if y<len(vocab.itos) else ents[y-len(vocab.itos)] for j,y in enumerate(x)])
        if "<eos>" in s: s = s.split("<eos>")[0]
        return s

    def build_out_text_for_document(self, doc_sentences):
        out_text = []
        global_idx = -1
        for sentence in doc_sentences:
            current_ent = None
            ner_pointer = 0
            if sentence['ner']:
                current_ent = (
                    sentence['ner'][ner_pointer][0],
                    sentence['ner'][ner_pointer][1],
                    f"<{sentence['ner'][ner_pointer][2].replace(' ', '').lower()}>"
                )

            for word in sentence['sentence']:
                global_idx += 1

                # Update current ent if necessary
                if current_ent and global_idx > current_ent[1]:
                    ner_pointer += 1

                    if len(sentence['ner']) > ner_pointer:
                        current_ent = (
                            sentence['ner'][ner_pointer][0],
                            sentence['ner'][ner_pointer][1],
                            f"<{sentence['ner'][ner_pointer][2].replace(' ', '').lower()}>"
                        )
                    else:
                        current_ent = None

                if not current_ent or global_idx < current_ent[0]:
                    out_text.append(word)
                    continue

                if global_idx >= current_ent[0] and global_idx <= current_ent[1]:
                    # If this marks the end word of the span, append the type here
                    if global_idx == current_ent[1]:
                        out_text.append(current_ent[2])
                        continue

        return out_text
