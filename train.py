import sys
import os
import util
from torchtext import data
from document_dataset import DocumentDataset
from models.model import Model
import torch
from torch.nn import functional as F
from torch import nn
from optimizers import MultipleOptimizer

from torch.utils.tensorboard import SummaryWriter
import subprocess

# Graph Writer modules
from GraphWriter.models.newmodel import model as graph
from GraphWriter.pargs import pargs, dynArgs

global_step_batch = 0


def update_lr(o, args, epoch):
    if epoch % args.lrstep == 0:
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

    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

    # util.print_config(config)

    util.set_gpus(0)

    writer = SummaryWriter(log_dir=config['log_dir'])
    args = dynArgs(args)
    config.device = args.device
    dataset_wrapper = DocumentDataset(config, args)

    # Graph writer arguments
    args = dynArgs(args)

    if config['train_graph_for'] or config['train_both_for']:
        graph_model = graph(args, dataset_wrapper.config, dataset_wrapper)
    else:
        graph_model = None

    if config['train_sci_for'] or config['train_both_for']:
        model = Model(config, dataset_wrapper)
    else:
        model = None

    # Move models to gpu?
    model = model.to(args.device)

    data_iter = data.Iterator(
        dataset_wrapper.dataset,
        config.batch_size,
        # device=args.device,
        sort_key=lambda x: len(x.text_len),
        repeat=False,
        train=True
    )

    val_iter = data.Iterator(
        dataset_wrapper.val_dataset,
        config.batch_size,
        # device=args.device,
        sort_key=lambda x: len(x.text_len),
        repeat=False,
        train=False
    )

    optimziers = {}
    if model:
        sci_opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        optimziers['sci'] = sci_opt
    else:
        sci_opt = None
    if graph_model:
        graph_opt = torch.optim.SGD(graph_model.parameters(), lr=args.lr, momentum=0.9)
        optimziers['graph'] = graph_opt
    else:
        graph_opt = None


    # Combined graph and sci optimizers
    optimizer = MultipleOptimizer(optimziers)

    if sci_opt:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=sci_opt, gamma=config["decay_rate"])

    offset = 0
    for epoch in range(max(config['train_graph_for'], config['train_sci_for']) + config['train_both_for']):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        if config['train_sci_for'] and config['train_sci_for'] > epoch:
            train_sci = True
        else:
            train_sci = False

        if config['train_graph_for'] and config['train_graph_for'] > epoch:
            train_graph = True
        else:
            train_graph = False

        if not train_graph and not train_sci:
            train_joint = True
        else:
            train_joint = False

        predict_dict, loss, offset = train(
            model,
            graph_model,
            dataset_wrapper,
            optimizer,
            writer,
            data_iter,
            args.device,
            config,
            train_sci or train_joint,
            train_graph or train_joint,
            offset
        )

        val_loss, val_sci_loss, val_gr_loss = evaluate(
            model,
            graph_model,
            dataset_wrapper,
            val_iter,
            args.device,
            config,
            train_sci or train_joint,
            train_graph or train_joint
        )

        if args.lrwarm and graph_opt:
            update_lr(graph_opt, args, epoch)

        print(f"epoch: {epoch + 1} - loss: {loss}")
        print(f"epoch: {epoch + 1} - VAL loss: {val_loss}")

        print("Saving models")

        if model:
            torch.save(
                model.state_dict(),
                f"{config['log_dir']}/model__{epoch + 1}.loss-{loss}.lr-{str(sci_opt.param_groups[0]['lr'])}"
            )

        # TODO
        if graph_model:
            torch.save(
                graph_model.state_dict(),
                f"{config['log_dir']}/graph_model__{epoch + 1}.loss-{loss}.lr-{str(graph_opt.param_groups[0]['lr'])}"
            )

        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/sci_loss', val_sci_loss, epoch)
        writer.add_scalar('val/gr_loss', val_gr_loss, epoch)

        scheduler.step()


def train(model, graph_model, dataset, optimizer, writer, data_iter, device, config, offset=0, train_graph=True, train_sci=True, train_joint=False):
    print("Training", end="\t")
    l = 0
    ex = 0

    for count, batch in enumerate(data_iter):
        print("Training sci batch")
        print(torch.cuda.memory_stats(device=device))
        print(f"Batch text length: {batch.text_len}")
        batch = dataset.fix_batch(batch)

        if train_joint:
            model.set_train_disjoint(False)
            graph_model.set_train_disjoint(False)

        if train_sci:
            predict_dict, sci_loss = model(batch)
        else:
            sci_loss = torch.tensor(100000)
            predict_dict = None

        if train_graph:
            p, planlogits = graph_model(batch)

            p = p[:, :-1, :].contiguous()

            tgt = batch.out[0][:, 1:].contiguous().view(-1).to(device)
            gr_loss = F.nll_loss(p.contiguous().view(-1, p.size(2)), tgt, ignore_index=1)
        else:
            gr_loss = torch.tensor(100000)

        if train_graph:
            total_loss = config['graph_writer_weight'] * gr_loss + config['scierc_weight'] * sci_loss
        else:
            total_loss = sci_loss

        optimizer.zero_grad()

        total_loss.backward()

        step_list = []
        if train_graph:
            step_list.append('sci')

        if train_sci:
            step_list.append('graph')

        optimizer.step(step_list)

        l += total_loss.item() * len(batch.doc_len)

        # Number of documents
        ex += len(batch.doc_len)

        # Summarize results
        step = count + offset

        writer.add_scalar('t/sci_loss/batch', sci_loss.item(), step)
        writer.add_scalar('t/gr_loss/batch', gr_loss.item(), step)
        writer.add_scalar('t/total/batch', total_loss.item(), step)

        # Zero gradients, perform a backward pass, and update params for the model1
        # optimizer.zero_grad()
        # sci_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), dataset.config["max_gradient_norm"])
        # optimizer.step()

        if train_sci:
            nn.utils.clip_grad_norm_(model.parameters(), dataset.config["max_gradient_norm"])
        if train_graph:
            nn.utils.clip_grad_norm_(graph_model.parameters(), args.clip)

    return predict_dict, l / (ex), step


def evaluate(model, graph_model, dataset, data_iter, device, config, train_graph=True):
    print("Evaluating", end="\t")
    if train_graph:
        graph_model.eval()
    model.eval()

    l = 0
    ex = 0
    sci_loss = 0
    gr_loss = 0
    count = 1
    for count, batch in enumerate(data_iter):
        with torch.no_grad():
            batch = dataset.fix_batch(batch)
            predict_dict, sci_loss = model(batch)

            if train_graph:
                p, planlogits = graph_model(batch)

                p = p[:, :-1, :].contiguous()

                tgt = batch.out[0][:, 1:].contiguous().view(-1).to(device)

                gr_loss = F.nll_loss(p.contiguous().view(-1, p.size(2)), tgt, ignore_index=1)

                total_loss = config['graph_writer_weight'] * gr_loss.item() + config['scierc_weight'] * sci_loss.item()
            else:
                total_loss = sci_loss
                gr_loss = torch.tensor(100000)

            l += total_loss * len(batch.doc_len)

            ex += len(batch.doc_len)

    # Summarize results
    model.train()
    if train_graph:
        graph_model.train()

    return l / ex, sci_loss / count, gr_loss / count


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])

    return float(result)

if __name__ == "__main__":
    args = pargs()
    main(args)
