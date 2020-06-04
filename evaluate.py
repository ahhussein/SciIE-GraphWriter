import sys
import os
import util
import data_utils
from torchtext import data
from document_dataset import DocumentDataset
from models.model import Model
import torch
import span_prune_cpp


def main():
    # ds = dataset(args)
    if len(sys.argv) > 1:
        name = sys.argv[1]
        print(f"Running experiment: {name} (from command-line argument).")

    else:
        name = "scientific_best_ner"
        print(f"Running experiment: {name} (from local variable).")

    config = util.get_config("experiments.conf")[name]
    report_frequency = config["report_frequency"]

    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

    # Dynamic batch size.
    config["batch_size"] = -1
    config["max_tokens_per_batch"] = -1

    # Use dev lm, if provided.
    if config["lm_path"] and "lm_path_dev" in config and config["lm_path_dev"]:
        config["lm_path"] = config["lm_path_dev"]

    # TODO test data set
    dataset = DocumentDataset(config)

    # TODO is training model eval
    model = Model(config, dataset)

    evaluator = 0 #LSGNEvaluator(config)

    # TODO log
    log_dir = config["log_dir"]

    max_f1 = 0
    best_task_f1 = {}

    # TODO multiple checkpoints
    model.load_state_dict(torch.load(log_dir))




if __name__ == "__main__":
    util.set_gpus()

    main()