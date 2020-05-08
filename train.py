import sys
import os
import util
from torchtext import data
from document_dataset import DocumentDataset
from models.model import Model


def main():
    # ds = dataset(args)
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "scientific_best_ner"
        #raise Exception('Experiment name has to be provided')

    config = util.get_config("experiments.conf")[name]

    #config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

    #util.print_config(config)
    #print(os.environ)

    util.set_gpus(0)

    dataset = DocumentDataset(config)

    # TODO is training
    #model = Model(config, 1)

    data_iter = data.Iterator(
        dataset,
        config.batch_size,
        #device=args.device,
        sort_key=lambda x:len(x['text_len']),
        repeat=False,
        train=True
    )

    for batch in data_iter:
        batch = dataset.fix_batch(batch)

if __name__ == "__main__":
    main()