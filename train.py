import sys
import os
import util
from torchtext import data
from document_dataset import TrainDataset
from models.model import Model
import torch
from torch.nn import functional as F
from torch import nn
from optimizers import MultipleOptimizer

from torch.utils.tensorboard import SummaryWriter

# Graph Writer modules
from GraphWriter.models.newmodel import model as graph
from GraphWriter.pargs import pargs,dynArgs

global_step_batch = 0

def update_lr(o,args,epoch):
  if epoch%args.lrstep == 0:
    o.param_groups[0]['lr'] = args.lrhigh
  else:
    o.param_groups[0]['lr'] -= args.lrchange

def main(args):
    # ds = dataset(args)
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "scientific_best_ner"

    # TODO-new integrate the two configs
    config = util.get_config("experiments.conf")[name]
    report_frequency = config["report_frequency"]

    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

    #util.print_config(config)

    util.set_gpus(0)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=config['log_dir'])

    # TODO test data set
    dataset_wrapper = TrainDataset(config)

    # Graph writer arguments
    args = dynArgs(args)
    graph_model = graph(args, dataset_wrapper.config)

    # TODO is training
    model = Model(config, dataset_wrapper)


    # Move models to gpu?
    # m = MODEL.to(args.device)

    # TODO early stopping?

    data_iter = data.Iterator(
        dataset_wrapper.dataset,
        config.batch_size,
        # device=args.device,
        sort_key=lambda x: len(x['text_len']),
        repeat=False,
        train=True
    )

    sci_opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    graph_opt = torch.optim.SGD(graph_model.parameters(), lr=args.lr, momentum=0.9)

    # Combined graph and sci optimizers
    optimizer = MultipleOptimizer(
        sci_opt,
        graph_opt
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=sci_opt, gamma=config["decay_rate"])

    # TEST - report_frequency = 10
    offset = 0
    for epoch in range(100):
        predict_dict, loss, offset = train(
            model,
            graph_model,
            dataset_wrapper,
            optimizer,
            writer,
            data_iter,
            args.device,
            config,
            offset
        )

        if epoch % report_frequency == 0:
            print(f"epoch: {epoch+1} - loss: {loss}")
            torch.save(model.state_dict(), f"{config['log_dir']}/model__{epoch+1}")

        # TODO validation loss
        vloss = 0
        if args.lrwarm:
            update_lr(graph_opt, args, epoch)
            print("Saving model")
            torch.save(
                graph_model.state_dict(),
                args.save + "/" + str(epoch) + ".vloss-" + vloss + ".lr-" + str(graph_opt.param_groups[0]['lr'])
            )

        writer.add_scalar('Loss/train', loss, epoch)
        print(f"epoch: {epoch + 1} - loss: {loss}")

        scheduler.step()

def train(model, graph_model, dataset, optimizer, writer, data_iter, device, config, offset = 0):
    l = 0
    ex = 0
    for count, batch in enumerate(data_iter):
        batch = dataset.fix_batch(batch)

        predict_dict, sci_loss = model(batch)

        p, planlogits = graph_model(batch)

        p = p[:, :-1, :].contiguous()

        tgt = batch.out[0][:, 1:].contiguous().view(-1).to(device)
        gr_loss = F.nll_loss(p.contiguous().view(-1, p.size(2)), tgt, ignore_index=1)

        total_loss = config['graph_writer_weight'] * gr_loss + config['scierc_weight'] * sci_loss

        total_loss.backward()

        l += total_loss.item() * len(batch.doc_len)

        #gr_loss.backward()

        # Clip gw parameters
        #nn.utils.clip_grad_norm_(graph_model.parameters(), args.clip)
        #loss += gr_loss.item() * len(batch.doc_len)

        optimizer.step()
        optimizer.zero_grad()

        # Number of documents
        ex += len(batch.doc_len)

        # Summarize results
        step = count + offset
        writer.add_scalar('Loss/sci_loss/batch', sci_loss, step=step)
        writer.add_scalar('Loss/gr_loss/batch', gr_loss, step=step)
        writer.add_scalar('Loss/total/batch', total_loss, step=step)

        # Zero gradients, perform a backward pass, and update params for the model1
        #optimizer.zero_grad()
        #sci_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), dataset.config["max_gradient_norm"])
        #optimizer.step()

        nn.utils.clip_grad_norm_(model.parameters(), dataset.config["max_gradient_norm"])
        nn.utils.clip_grad_norm_(graph_model.parameters(), args.clip)

    return predict_dict, l/(ex), step

if __name__ == "__main__":
    args = pargs()
    main(args)