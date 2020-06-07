import sys
import os
import util
from document_dataset import EvalDataset
from models.model import Model
import torch
from evaluator import Evaluator
from eval_iter import EvalIterator



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
    print(config['output_path'])

    # Dynamic batch size.
    config["batch_size"] = -1
    config["max_tokens_per_batch"] = -1

    # Use dev lm, if provided.
    if config["lm_path"] and "lm_path_dev" in config and config["lm_path_dev"]:
        config["lm_path"] = config["lm_path_dev"]

    # TODO test data set
    dataset = EvalDataset(config=config)

    # TODO is training model eval
    model = Model(config, dataset)

    evaluator = Evaluator(config, dataset, model)

    # TODO log
    log_dir = config["log_dir"]

    model.load_state_dict(torch.load(f"{log_dir}/model__1"))

    # Load batch of sentences for each document
    data_iter = EvalIterator(dataset, batch_size=1, sort=False, train=False)

    for count, batch in enumerate(data_iter):
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