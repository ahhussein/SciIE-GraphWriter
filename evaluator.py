import datetime

import data_utils
import time
import coref_metrics
import debug_utils
import operator
import inference_utils
import util
import json
from encoder import MyEncoder

class Evaluator:
    def __init__(self, config, dataset, logger = None):
        self.config = config
        self.dataset = dataset
        self.eval_data = self.dataset.eval_data
        self.coref_eval_data = self.dataset.coref_eval_data
        #self.debug_printer = debug_utils.DebugPrinter()
        self.start_time = time.time()
        self.ner_predictions = []
        self.gold_ner = []
        self.rel_predictions = []
        self.gold_rels = []
        self.coref_predictions = {}
        self.mention_evaluators = {k: data_utils.RetrievalEvaluator() for k in [-3, -2, -1, 10, 20, 30, 40, 50]}
        self.entity_evaluators = {k: data_utils.RetrievalEvaluator() for k in [-3, -2, -1, 10, 20, 30, 40, 50, 70]}
        self.coref_evaluator = coref_metrics.CorefEvaluator()
        self.total_loss = 0
        self.json_data = []
        self.logger = logger


    def evaluate(self, predictions, total_loss):
        self.total_loss = total_loss
        for doc_key, doc_example in self.coref_eval_data.items():
            json_output = {'doc_key': doc_key}

            sentences = doc_example["sentences"]

            decoded_predictions = inference_utils.mtl_decode(
                sentences, predictions[doc_key], self.dataset.ner_labels_inv,
                self.dataset.rel_labels_inv, self.config
            )


            # Relation extraction.
            if "rel" in decoded_predictions:
                self.rel_predictions.extend(decoded_predictions["rel"])
                goldrels = [gsentence[3] for gsentence in [gsentences for gsentences in self.eval_data[doc_key]]]
                #goldrels_flattened = [item for sublist in goldrels for item in sublist]

                self.gold_rels.extend(goldrels)
                json_output['relation'] = decoded_predictions["rel"]

                rel_sent_id = 0
                for j in range(len(sentences)):
                    sent_example = self.eval_data[doc_key][rel_sent_id][3]  # sentence, srl, ner, relations
                    text_length = len(sentences[j])
                    ne = predictions[doc_key]["num_entities"][j]
                    gold_entities = set([])
                    for rel in sent_example:
                        gold_entities.update([rel[:2], rel[2:4]])

                    data_utils.evaluate_retrieval(
                        predictions[doc_key]["candidate_starts"][j], predictions[doc_key]["candidate_ends"][j],
                        predictions[doc_key]["candidate_entity_scores"][j], predictions[doc_key]["entity_starts"][j][:ne],
                        predictions[doc_key]["entity_ends"][j][:ne], gold_entities, text_length, self.entity_evaluators)
                    rel_sent_id += 1

            if "ner" in decoded_predictions:
                self.ner_predictions.extend(decoded_predictions["ner"])
                goldners = [gsentence[2] for gsentence in [gsentences for gsentences in self.eval_data[doc_key]]]
                #goldners_flattened = [item for sublist in goldners for item in sublist]

                self.gold_ner.extend(goldners)
                json_output['ner'] = decoded_predictions["ner"]

            if "predicted_clusters" in decoded_predictions:
                gold_clusters = [tuple(tuple(m) for m in gc) for gc in doc_example["clusters"]]
                gold_mentions = set([])
                mention_to_gold = {}
                for gc in gold_clusters:
                    for mention in gc:
                        mention_to_gold[mention] = gc
                        gold_mentions.add(mention)
                self.coref_evaluator.update(decoded_predictions["predicted_clusters"], gold_clusters,
                                       decoded_predictions["mention_to_predicted"],
                                       mention_to_gold)
                self.coref_predictions[doc_example["doc_key"]] = decoded_predictions["predicted_clusters"]
                json_output['coref'] = decoded_predictions["predicted_clusters"]

                # Evaluate retrieval.
                doc_text_length = sum([len(s) for s in sentences])
                data_utils.evaluate_retrieval(
                    predictions[doc_key]["candidate_mention_starts"], predictions[doc_key]["candidate_mention_ends"],
                    predictions[doc_key]["candidate_mention_scores"], predictions[doc_key]["mention_starts"],
                    predictions[doc_key]["mention_ends"],
                    gold_mentions, doc_text_length, self.mention_evaluators)


            self.json_data.append(json_output)


    def summarize_results(self):
        def _k_to_tag(k):
            if k == -3:
                return "oracle"
            elif k == -2:
                return "actual"
            elif k == -1:
                return "exact"
            elif k == 0:
                return "threshold"
            else:
                return "{}%".format(k)


        #self.debug_printer.close()
        summary_dict = {}
        task_to_f1 = {}  # From task name to F1.
        elapsed_time = time.time() - self.start_time

        data = []
        for doc_id, value in self.eval_data.items():
            data.extend(value)

        sentences, gold_srl, gold_ner, gold_relations = zip(*data)

        # Summarize results.
        if self.config["relation_weight"] > 0:
            precision, recall, f1 = (
                data_utils.compute_relation_f1(self.gold_rels, self.rel_predictions)
            )

            task_to_f1["relations"] = f1
            summary_dict["Relation F1"] = f1
            summary_dict["Relation precision"] = precision
            summary_dict["Relation recall"] = recall
            for k, evaluator in sorted(self.entity_evaluators.items(), key=operator.itemgetter(0)):
                tags = ["{} {} @ {}".format("Entities", t, _k_to_tag(k)) for t in ("R", "P", "F")]
                results_to_print = []
                for t, v in list(zip(tags, evaluator.metrics())):
                    results_to_print.append("{:<10}: {:.4f}".format(t, v))
                    summary_dict[t] = v
                self.log("info",", ".join(results_to_print))

        if self.config["ner_weight"] > 0:
            ner_precision, ner_recall, ner_f1, ul_ner_prec, ul_ner_recall, ul_ner_f1, ner_label_mat = (
                data_utils.compute_span_f1(self.gold_ner, self.ner_predictions, "NER"))
            summary_dict["NER F1"] = ner_f1
            summary_dict["NER precision"] = ner_precision
            summary_dict["NER recall"] = ner_recall
            summary_dict["Unlabeled NER F1"] = ul_ner_f1
            summary_dict["Unlabeled NER precision"] = ul_ner_prec
            summary_dict["Unlabeled NER recall"] = ul_ner_recall

            # Write NER prediction to IOB format and run official eval script.
            util.print_to_iob2(sentences, self.gold_ner, self.ner_predictions, self.config["ner_conll_eval_path"])
            task_to_f1["ner"] = ner_f1

        if self.config["coref_weight"] > 0:
            p, r, f = self.coref_evaluator.get_prf()
            summary_dict["Average Coref F1 (py)"] = f
            self.log("info", "Average F1 (py): {:.2f}%".format(f * 100))
            summary_dict["Average Coref precision (py)"] = p
            self.log("info","Average precision (py): {:.2f}%".format(p * 100))
            summary_dict["Average Coref recall (py)"] = r
            self.log("info","Average recall (py): {:.2f}%".format(r * 100))

            task_to_f1["coref"] = f * 100  # coref_conll_f1
            for k, evaluator in sorted(self.mention_evaluators.items(), key=operator.itemgetter(0)):
                tags = ["{} {} @ {}".format("Mentions", t, _k_to_tag(k)) for t in ("R", "P", "F")]
                results_to_print = []
                for t, v in list(zip(tags, evaluator.metrics())):
                    results_to_print.append("{:<10}: {:.4f}".format(t, v))
                    summary_dict[t] = v
                self.log("info",", ".join(results_to_print))

        summary_dict["Dev Loss"] = self.total_loss / len(self.coref_eval_data)

        self.log("info","Decoding took {}.".format(str(datetime.timedelta(seconds=int(elapsed_time)))))
        self.log("info",
            "Decoding speed: {}/document, or {}/sentence.".format(
                str(datetime.timedelta(seconds=int(elapsed_time / len(self.coref_eval_data)))),
                str(datetime.timedelta(seconds=int(elapsed_time / len(self.eval_data))))
            )
        )

        metric_names = self.config["main_metrics"].split("_")
        main_metric = sum([task_to_f1[t] for t in metric_names]) / len(metric_names)
        self.log("info","Combined metric ({}): {}".format(self.config["main_metrics"], main_metric))

        return summary_dict, main_metric, task_to_f1

    def write_out(self):
        outfn = self.config["output_path"]
        print('writing to ' + outfn)
        with open(outfn, 'w') as f:
            for json_line in self.json_data:
                f.write(json.dumps(json_line, cls=MyEncoder))
                f.write('\n')

    def log(self, level, message):
        if self.logger:
            getattr(self.logger, level)(message)





