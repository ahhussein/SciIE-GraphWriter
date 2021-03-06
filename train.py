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
import logging
from models.vertex_embeddings import VertexEmbeddings

logger = logging.getLogger('myapp')

# Graph Writer modules
from GraphWriter.models.newmodel import model as graph
from GraphWriter.pargs import pargs, dynArgs

global_step_batch = 0

def set_logger(log_file):
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)

def update_lr(o, args, epoch):
    if epoch % args.lrstep == 0:
        o.param_groups[0]['lr'] = args.lrhigh
    else:
        o.param_groups[0]['lr'] -= args.lrchange

def train_sci(model, graph_model, dataset_wrapper, writer, config, device, optimizer, sci_opt, scheduler):
    # joint training
    data_iter = data.Iterator(
        dataset_wrapper.dataset,
        config.batch_size,
        device=device,
        sort_key=lambda x: sum(x.text_len),
        repeat=False,
        train=True
    )

    offset = 0
    for epoch in range(config['train_sci_for']):
        predict_dict, loss, sci_loss, gr_loss, offset = train(
            model,
            graph_model,
            dataset_wrapper,
            optimizer,
            writer,
            config,
            data_iter,
            offset,
            False,
            True,
            False
        )

        logger.info(f"epoch Sci: {epoch + 1} - loss: {sci_loss}")

        logger.info("Saving models")

        torch.save(
            model.state_dict(),
            f"{config['log_dir']}/model__{epoch + 1}.loss-{loss}.lr-{str(sci_opt.param_groups[0]['lr'])}"
        )

        writer.add_scalar('train/sci_loss', sci_loss, epoch)

        scheduler.step()

def train_graph(model, graph_model, dataset_wrapper, writer, config, device, optimizer, graph_opt):
    # joint training
    data_iter = data.Iterator(
        dataset_wrapper.dataset,
        config.batch_size,
        device=device,
        sort_key=lambda x: sum(x.text_len),
        repeat=False,
        train=True
    )

    val_iter = data.Iterator(
        dataset_wrapper.val_dataset,
        config.batch_size,
        device=device,
        sort_key=lambda x: sum(x.text_len),
        repeat=False,
        train=False
    )

    # Train the graph
    # Freeze vertex embs
    for param in graph_model.vertex_embeddings.parameters():
        param.requires_grad = False

    offset = 0
    for epoch in range(config['train_graph_for']):
        predict_dict, loss, sci_loss, gr_loss, offset = train(
            model,
            graph_model,
            dataset_wrapper,
            optimizer,
            writer,
            config,
            data_iter,
            offset,
            True,
            False,
            False
        )

        # val_loss, val_sci_loss, val_gr_loss = evaluate(
        #     model,
        #     graph_model,
        #     dataset_wrapper,
        #     config,
        #     val_iter,
        #     True,
        #     False,
        #     False
        # )

        if args.lrwarm and graph_opt:
            update_lr(graph_opt, args, epoch)

        logger.info(f"epoch graph: {epoch + 1} - loss: {gr_loss}")
        #logger.info(f"epoch graph: {epoch + 1} - VAL loss: {val_loss}")

        logger.info("Saving models")

        torch.save(
            graph_model.state_dict(),
            f"{config['log_dir']}/graph_model__{epoch + 1}.loss-{loss}.lr-{str(graph_opt.param_groups[0]['lr'])}"
        )

        writer.add_scalar('train/gr_loss', gr_loss, epoch)
        #writer.add_scalar('val/gr_loss', val_loss, epoch)

def train_joint(model, graph_model, dataset_wrapper, writer, config, device, optimizer, graph_opt, sci_opt, scheduler):
    offset = 0

    # Freeze the attention params
    for param in graph_model.parameters():
        param.requires_grad = False

    for param in graph_model.vertex_embeddings.parameters():
        param.requires_grad = True

    # joint training
    data_iter = data.Iterator(
        dataset_wrapper.dataset,
        config.batch_size_joint,
        device=device,
        sort_key=lambda x: sum(x.text_len),
        repeat=False,
        train=True
    )

    val_iter = data.Iterator(
        dataset_wrapper.val_dataset,
        config.batch_size_joint,
        device=device,
        sort_key=lambda x: sum(x.text_len),
        repeat=False,
        train=False
    )

    for epoch in range(config['train_both_for']):
        predict_dict, loss, sci_loss, gr_loss, offset = train(
            model,
            graph_model,
            dataset_wrapper,
            optimizer,
            writer,
            config,
            data_iter,
            offset,
            False,
            False,
            True
        )

        val_loss, val_sci_loss, val_gr_loss = evaluate(
            model,
            graph_model,
            dataset_wrapper,
            config,
            val_iter,
            False,
            False,
            True
        )

        if args.lrwarm and graph_opt:
            update_lr(graph_opt, args, epoch)

        logger.info(f"epoch joint: {epoch + 1} - loss: {loss}")
        logger.info(f"epoch joint: {epoch + 1} - VAL loss: {val_loss}")

        logger.info("Saving models")

        torch.save(
            model.state_dict(),
            f"{config['log_dir']}/model__joint_{epoch + 1}.loss-{loss}.lr-{str(sci_opt.param_groups[0]['lr'])}"
        )

        torch.save(
            graph_model.state_dict(),
            f"{config['log_dir']}/graph_model__joint_{epoch + 1}.loss-{loss}.lr-{str(graph_opt.param_groups[0]['lr'])}"
        )

        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/gr_loss', gr_loss, epoch)
        writer.add_scalar('train/sci_loss', sci_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/sci_loss', val_sci_loss, epoch)
        writer.add_scalar('val/gr_loss', val_gr_loss, epoch)

        scheduler.step()

def main(args):
    # ds = dataset(args)
    args = dynArgs(args)
    config = util.get_config("experiments.conf")[args.exp]

    set_logger(args.logfile)
    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], args.logdir))

    util.set_gpus(0)

    writer = SummaryWriter(log_dir=config['log_dir'])
    config.device = args.device

    dataset_wrapper = DocumentDataset(config, args)
    vertex_embeddings = VertexEmbeddings(config, dataset_wrapper)

    # Graph writer arguments
    if config['train_graph_for'] or config['train_both_for']:
        graph_model = graph(args, config, dataset_wrapper, vertex_embeddings, logger)
        # Move models to gpu?
        graph_model = graph_model.to(args.device)

    else:
        graph_model = None

    if config['train_sci_for'] or config['train_both_for']:
        model = Model(config, dataset_wrapper, vertex_embeddings, logger)
        model = model.to(args.device)
    else:
        model = None

    optimziers = {}
    if model:
        sci_opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        optimziers['sci'] = sci_opt

        max_model_checkpoint = os.path.join(config["log_dir"], "model.max")
        if os.path.exists(max_model_checkpoint):
            model.load_state_dict(torch.load(max_model_checkpoint))
            logger.info("Scierc Model Loaded")

    else:
        sci_opt = None
    if graph_model:
        graph_opt = torch.optim.SGD(graph_model.parameters(), lr=args.lr, momentum=0.9)
        optimziers['graph'] = graph_opt

        max_model_checkpoint = os.path.join(config["log_dir"], "graph_model.max")
        if os.path.exists(max_model_checkpoint):
            graph_model.load_state_dict(torch.load(max_model_checkpoint))
            logger.info("Graph Model Loaded")
    else:
        graph_opt = None

    logger.info(f"Training SCIERC for: {config['train_sci_for']} epochs")
    logger.info(f"Training Graph for: {config['train_graph_for']} epochs")
    logger.info(f"Training Jointly for: {config['train_both_for']} epochs")
    logger.info(f"Batch size: {config.batch_size}")

    # Combined graph and sci optimizers
    optimizer = MultipleOptimizer(optimziers)

    if sci_opt:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=sci_opt, gamma=config["decay_rate"])

    torch.autograd.set_detect_anomaly(True)

    if args.device.type != 'cpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if config['train_sci_for'] > 0:
        train_sci(
            model,
            graph_model,
            dataset_wrapper,
            writer,
            config,
            args.device,
            optimizer,
            sci_opt,
            scheduler
        )

    if config['train_graph_for'] > 0:
        train_graph(
            model,
            graph_model,
            dataset_wrapper,
            writer,
            config,
            args.device,
            optimizer,
            graph_opt
        )

    if config['train_both_for'] > 0:
        train_joint(
            model,
            graph_model,
            dataset_wrapper,
            writer,
            config,
            args.device,
            optimizer,
            graph_opt,
            sci_opt,
            scheduler
        )


def train(model, graph_model, dataset, optimizer, writer, config, data_iter, offset=0, train_graph=True, train_sci=True, train_joint=False):
    logger.info("Training")

    l = 0
    ex = 0
    g_loss = 0
    sc_loss = 0
    total_loss = torch.tensor(0)
    for count, batch in enumerate(data_iter):
        batch = dataset.fix_batch(batch)

        if train_joint:
            logger.info("Training Joint")
            model.set_train_disjoint(False)
            graph_model.set_train_disjoint(False)

        if train_sci or train_joint:
            try:
                logger.info(f"SCI Batch: {count}")
                predict_dict, sci_loss = model(batch)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.error('| WARNING: ran out of memory, skipping SCI batch')
                    logger.info(f"Batch sizes: {batch.text_len}")
                    logger.info(f"Batch key: {batch.doc_key}")
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                else:
                    raise e

                # skip batch
                continue
        else:
            sci_loss = torch.tensor(0)
            predict_dict = None

        if train_graph or train_joint:
            try:
                logger.info(f"Graph Batch: {count}")
                p, planlogits = graph_model(batch)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.error('| WARNING: ran out of memory, skipping GRAPH batch')
                    logger.info(f"Batch sizes: {batch.text_len}")
                    logger.info(f"Batch key: {batch.doc_key}")

                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                else:
                    raise e
                # skip batch
                continue

            p = p[:, :-1, :].contiguous()

            tgt = batch.tgt[:, 1:].contiguous().view(-1).to(args.device)
            gr_loss = F.nll_loss(p.contiguous().view(-1, p.size(2)), tgt, ignore_index=1)
        else:
            gr_loss = torch.tensor(0)


        optimizer.zero_grad()

        if train_joint:
            total_loss = config['graph_writer_weight'] * gr_loss + config['scierc_weight'] * sci_loss
            total_loss.backward()
            l += total_loss.item() * len(batch.doc_len)
        if train_graph and not train_joint:
            gr_loss.backward()
            g_loss += gr_loss.item() * len(batch.doc_len)
        if train_sci and not train_joint:
            sci_loss.backward()
            sc_loss += sci_loss.item() * len(batch.doc_len)

        # Summarize results
        step = count + offset

        step_list = []
        if train_graph or train_joint:
            step_list.append('graph')

        if train_sci or train_joint:
            step_list.append('sci')

        if train_sci or train_joint:
            nn.utils.clip_grad_norm_(model.parameters(), dataset.config["max_gradient_norm"])
        if train_graph or train_joint:
            nn.utils.clip_grad_norm_(graph_model.parameters(), args.clip)

        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram('params/' + name, param.clone().detach().cpu().numpy(), step, bins=20)
                writer.add_histogram('grads/' + name, param.grad.clone().detach().cpu().numpy(), step, bins=20)


        optimizer.step(step_list)

        # Number of documents
        ex += len(batch.doc_len)

        writer.add_scalar('t/sci_loss/batch', sci_loss.item(), step)
        writer.add_scalar('t/gr_loss/batch', gr_loss.item(), step)
        writer.add_scalar('t/total/batch', total_loss.item(), step)

        # Zero gradients, perform a backward pass, and update params for the model1
        # optimizer.zero_grad()
        # sci_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), dataset.config["max_gradient_norm"])
        # optimizer.step()


    return predict_dict, l / (ex), sc_loss / (ex), g_loss / (ex), step


def evaluate(model, graph_model, dataset, config, data_iter, train_graph=True, train_sci=True, train_joint=False):
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
            if train_joint:
                model.set_train_disjoint(False)
                graph_model.set_train_disjoint(False)

            if train_sci or train_joint:
                predict_dict, sci_loss = model(batch)
            else:
                sci_loss = torch.tensor(0)
                predict_dict = None

            if train_graph or train_joint:
                p, planlogits = graph_model(batch)

                p = p[:, :-1, :].contiguous()

                tgt = batch.tgt[:, 1:].contiguous().view(-1).to(args.device)
                gr_loss = F.nll_loss(p.contiguous().view(-1, p.size(2)), tgt, ignore_index=1)
                if ex == 0:
                    g = p[0].max(1)[1]
                    #print(ds.reverse(g, b.rawent[0]))
            else:
                gr_loss = torch.tensor(0)

            if train_sci:
                total_loss = sci_loss.item()
            elif train_graph:
                total_loss = gr_loss.item()
            else:
                total_loss = config['graph_writer_weight'] * gr_loss.item() + config['scierc_weight'] * sci_loss.item()

            l += total_loss * len(batch.doc_len)

            ex += len(batch.doc_len)

    # Summarize results
    if model:
        model.train()
    if graph_model:
        graph_model.train()

    return l / ex, sci_loss.item() / ex, gr_loss.item() / ex


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])

    return float(result)

# def plot_grad_flow(named_parameters, writer):
#     '''Plots the gradients flowing through different layers in the net during training.
#     Can be used for checking for possible gradient vanishing / exploding problems.
#
#     Usage: Plug this function in Trainer class after loss.backwards() as
#     "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
#     ave_grads = []
#     max_grads = []
#     layers = []
#     for n, p in named_parameters:
#         if (p.requires_grad) and ("bias" not in n) and (p.grad != None):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean())
#             max_grads.append(p.grad.abs().max())
#             writer.add_scalars(n, {
#                 'avg': p.grad.abs().mean(),
#                 'max': p.grad.abs().max(),
#             })
#
#     return ave_grads, max_grads, layers

if __name__ == "__main__":
    args = pargs()
    main(args)
