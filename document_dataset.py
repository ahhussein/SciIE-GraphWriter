from torch.utils.data import Dataset
import json
import util
import data_utils
import random
import h5py


class DocumentDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.lm_file = h5py.File(self.config["lm_path"], "r")
        self.lm_layers = self.config["lm_layers"]
        self.lm_size = self.config["lm_size"]

        self._load_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _load_data(self):
        with open(self.config["train_path"], "r", encoding='utf8') as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        self.populate_sentence_offset(train_examples)

        adaptive_batching = (self.config["max_tokens_per_batch"] > 0)

        self._read_documents(train_examples)


    def _read_documents(self, train_examples):
        random.shuffle(train_examples)

        # List of documents, each holds a list of sentences.
        doc_examples = []

        cluster_id_offset = 0
        num_sentences = 0
        num_mentions = 0

        for doc_id, document in enumerate(train_examples, 1):
            doc_examples.append([])

            # Read sentences in a document
            for example in self._split_document_example(document):
                example["doc_id"] = doc_id
                example["cluster_id_offset"] = cluster_id_offset
                doc_examples[-1].append(example)
                num_mentions += len(example["coref"])
            cluster_id_offset += len(document["clusters"])
            num_sentences += len(doc_examples[-1])

        print("Load {} training documents with {} sentences, {} clusters, and {} mentions.".format(
            doc_id, num_sentences, cluster_id_offset, num_mentions))

        for examples in doc_examples:
          tensor_examples = [self.tensorize_example(e, is_training=True) for e in examples]


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

    def tensorize_example(self, example, is_training):
        """
        Tensorize examples and cache embeddings.
        """
        sentence = example["sentence"]
        doc_key = example["doc_key"]
        sent_id = example["sent_id"]
        word_offset = example["word_offset"]
        text_len = len(sentence)

        lm_doc_key = None
        lm_sent_key = None
        transpose = True

        lm_doc_key = doc_key + "_" + str(sent_id)
        transpose = False

        # Load cached LM.
        lm_emb = data_utils.load_lm_embeddings_for_sentence(
            self.lm_file,
            lm_doc_key,
            lm_sent_key,
            transpose
        )

        # max_word_length = max(max(len(w) for w in sentence), max(self.config["filter_widths"]))
        #
        # # Preapre context word embedding for a sentence
        # context_word_emb = np.zeros([text_len, self.context_embeddings.size])
        # head_word_emb = np.zeros([text_len, self.head_embeddings.size])
        # char_index = np.zeros([text_len, max_word_length])
        # for j, word in enumerate(sentence):
        #     context_word_emb[j] = self.context_embeddings[word]
        #     head_word_emb[j] = self.head_embeddings[word]
        #     char_index[j, :len(word)] = [self.char_dict[c] for c in word]
        #
        # ner_starts, ner_ends, ner_labels = (
        #     tensorize_labeled_spans(example["ner"], self.ner_labels))
        # coref_starts, coref_ends, coref_cluster_ids = (
        #     tensorize_labeled_spans(example["coref"], label_dict=None))
        # # predicates, arg_starts, arg_ends, arg_labels = (
        # #     tensorize_srl_relations(example["srl"], self.srl_labels,
        # #     filter_v_args=self.config["filter_v_args"]))
        # rel_e1_starts, rel_e1_ends, rel_e2_starts, rel_e2_ends, rel_labels = (
        #     tensorize_entity_relations(example["relations"], self.rel_labels,
        #                                filter_reverse=self.config["filter_reverse_relations"]))
        #
        # # For gold predicate experiment.
        # # gold_predicates = get_all_predicates(example["srl"]) - word_offset
        # example_tensor = {
        #     # Inputs.
        #     "tokens": sentence,
        #     "context_word_emb": context_word_emb,
        #     "head_word_emb": head_word_emb,
        #     # Lm embeddings for a sentence
        #     "lm_emb": lm_emb,
        #     "char_idx": char_index,
        #     "text_len": text_len,
        #     "doc_id": example["doc_id"],
        #     "doc_key": example["doc_key"],
        #     "is_training": is_training,
        #     # "gold_predicates": gold_predicates,
        #     # "num_gold_predicates": len(gold_predicates),
        #     # Labels.
        #     # Word offset to the start of the document since ner_starts is just relevant to the start of the sentence
        #     "ner_starts": ner_starts - word_offset,
        #     "ner_ends": ner_ends - word_offset,
        #
        #     # Ids of the relations reelative to start and end
        #     "ner_labels": ner_labels,
        #     # "predicates": predicates - word_offset,
        #     # "arg_starts": arg_starts - word_offset,
        #     # "arg_ends": arg_ends - word_offset,
        #     # "arg_labels": arg_labels,
        #     "coref_starts": coref_starts - word_offset,
        #     "coref_ends": coref_ends - word_offset,
        #     "coref_cluster_ids": coref_cluster_ids + example["cluster_id_offset"],
        #     "rel_e1_starts": rel_e1_starts - word_offset,
        #     "rel_e1_ends": rel_e1_ends - word_offset,
        #     "rel_e2_starts": rel_e2_starts - word_offset,
        #     "rel_e2_ends": rel_e2_ends - word_offset,
        #     "rel_labels": rel_labels,
        #     # "srl_len": len(predicates),
        #     "ner_len": len(ner_starts),
        #     "coref_len": len(coref_starts),
        #     "rel_len": len(rel_e1_starts)
        # }
        # return example_tensor




