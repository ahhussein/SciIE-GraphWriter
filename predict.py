import sys
import os
import util
from document_dataset import DocumentDataset
from models.model import Model
import torch
from evaluator import Evaluator
from eval_iter import EvalIterator
from torchtext import data



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

    model = Model(config, dataset)

    evaluator = Evaluator(config, dataset, model)

    # TODO log
    log_dir = config["log_dir"]

    model.load_state_dict(torch.load(f"{log_dir}/model__7.loss-0.0.lr-0.000497007490007497"))

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
            evaluator.evaluate(doc_batch)
            if (count + 1) % 50 == 0:
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