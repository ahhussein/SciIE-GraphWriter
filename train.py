import sys
import os
import util
from document_dataset import DocumentDataset

def main():
    # ds = dataset(args)
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "scientific_best_ner"
        #raise Exception('Experiment name has to be provided')

    config = util.get_config("experiments.conf")[name]

    #config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

    util.print_config(config)
    print(os.environ)

    util.set_gpus(0)

    dataset = DocumentDataset(config)

if __name__ == "__main__":
    main()