import sys
import os
import util
from document_dataset import DocumentDataset
from models.model import Model
import torch
from evaluator import Evaluator
from GraphWriter.pargs import pargs, dynArgs
from torchtext import data
import logging
from models.span_embeddings_wrapper import SpanEmbeddingsWrapper
from torch import nn


torch.manual_seed(0)

logger = logging.getLogger('myapp')
hdlr = logging.FileHandler('predict.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)



def main():
    # ds = dataset(args)
    if len(sys.argv) > 1:
        name = sys.argv[1]
        print(f"Running experiment: {name} (from command-line argument).")

    else:
        name = "scientific_best_ner"
        print(f"Running experiment: {name} (from local variable).")

    config = util.get_config("experiments.conf")[name]

    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

    # Dynamic batch size.
    config["batch_size"] = -1
    config["max_tokens_per_batch"] = -1

    dataset = DocumentDataset(config=config, is_eval=True)

    embedding_wrapper = SpanEmbeddingsWrapper(config, dataset)
    rel_embs = nn.Embedding(2 * len(dataset.rel_labels_extended) - 1, 500)

    model = Model(config, dataset, embedding_wrapper, rel_embs, logger)

    model.to(args.device)

    evaluator = Evaluator(config, dataset, model, logger)

    # TODO log
    log_dir = config["log_dir"]

    model.load_state_dict(torch.load(f"{log_dir}/model__1.loss-0.0.lr-0.0005"))

    # Load batch of sentences for each document
    data_iter = data.Iterator(
        dataset.test_dataset,
        1,
        # device=args.device,
        sort_key=lambda x: len(x.text_len),
        repeat=False,
        train=False
    )

    for count, batch in enumerate(data_iter):
        with torch.no_grad():

            doc_batch = dataset.fix_batch(batch)
            logger.info(f"Batch size: {doc_batch.text_len}")
            logger.info(f"Batch key: {doc_batch.doc_key}")
            evaluator.evaluate(doc_batch)

            print("Evaluated {}/{} documents.".format(count + 1, len(evaluator.coref_eval_data)))
    evaluator.write_out()
    # Move to evaualtor
    # summary_dict, main_metric, task_to_f1 = evaluator.summarize_results()
    # print(summary_dict)
    # print(main_metric)
    # print(task_to_f1)






if __name__ == "__main__":
    util.set_gpus()

    main()