import sys
import os
import util
from torchtext import data
from document_dataset import TrainDataset
from models.model import Model
import torch
from torch.utils.tensorboard import SummaryWriter


def main():
    # ds = dataset(args)
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "scientific_best_ner"

    config = util.get_config("experiments.conf")[name]
    report_frequency = config["report_frequency"]

    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

    #util.print_config(config)

    util.set_gpus(0)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=config['log_dir'])

    # TODO test data set
    dataset = TrainDataset(config)

    # TODO is training
    model = Model(config, dataset)

    data_iter = data.Iterator(
        dataset,
        config.batch_size,
        # device=args.device,
        sort_key=lambda x: len(x['text_len']),
        repeat=False,
        train=True
    )

    # Get random input
    #inputs = dataset.fix_batch(next(iter(data_iter)))

    #writer.add_graph(model, inputs)

    # TODO clip gradients? see TF srl_model
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config["decay_rate"])

    # TEST - report_frequency = 10
    for epoch in range(100):
        predict_dict, loss = train(model, dataset, optimizer, writer, data_iter)

        if epoch % report_frequency == 0:
            print(f"epoch: {epoch+1} - loss: {loss}")
            torch.save(model.state_dict(), f"{config['log_dir']}/model__{epoch+1}")

        writer.add_scalar('Loss/train', loss, epoch)

        scheduler.step()

def train(model, dataset, optimizer, writer, data_iter):
    l = 0
    for count, batch in enumerate(data_iter):
        batch = dataset.fix_batch(batch)

        predict_dict, loss = model(batch)

        model.prepare_for_graph(predict_dict, batch)

        writer.add_scalar('Loss/batch', loss, count)

        # Zero gradients, perform a backward pass, and update params.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), dataset.config["max_gradient_norm"])
        optimizer.step()

        l += loss

    return predict_dict, l/(count+1)

if __name__ == "__main__":
    main()