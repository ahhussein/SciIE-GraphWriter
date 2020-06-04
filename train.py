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
    else:
        name = "scientific_best_ner"

    config = util.get_config("experiments.conf")[name]
    report_frequency = config["report_frequency"]

    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

    util.print_config(config)

    util.set_gpus(0)

    # TODO test data set
    dataset = DocumentDataset(config)

    # TODO is training
    model = Model(config, dataset)

    # TODO clip gradients? see TF srl_model
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config["decay_rate"])

    for epoch in range(20):
        predict_dict, loss = train(model, dataset, config, optimizer)

        if epoch % report_frequency == 0:
            print(epoch+1, loss)
            torch.save(model.state_dict(), config['log_dir'])

        scheduler.step()

def train(model, dataset, config, optimizer):
    data_iter = data.Iterator(
        dataset,
        config.batch_size,
        # device=args.device,
        sort_key=lambda x: len(x['text_len']),
        repeat=False,
        train=True
    )

    l = 0
    for count, batch in enumerate(data_iter):
        batch = dataset.fix_batch(batch)

        predict_dict, loss = model(batch)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l += loss

    return predict_dict, l/(count+1)

if __name__ == "__main__":
    main()