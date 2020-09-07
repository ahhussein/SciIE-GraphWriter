\
import glob
import sys
import os
import util
from document_dataset import DocumentDataset
from models.model import Model
import torch
from evaluator import Evaluator
from GraphWriter.pargs import pargs, dynArgs
from torchtext import data
import logging
from models.vertex_embeddings import VertexEmbeddings
import time
import shutil
import re

logger = logging.getLogger('myapp')
hdlr = logging.FileHandler('eval-sci.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)




def main(args):
    # ds = dataset(args)
    if len(sys.argv) > 1:
        name = sys.argv[1]
        print(f"Running experiment: {name} (from command-line argument).")

    else:
        name = "scientific_best_ner"
        print(f"Running experiment: {name} (from local variable).")

    config = util.get_config("experiments.conf")[name]

    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

    args = dynArgs(args)

    dataset = DocumentDataset(config, args, True)

    config.device = args.device

    # TODO ensure embeddings are loaded to the reference
    vertex_embeddings = VertexEmbeddings(config, dataset)

    model = Model(config, dataset, vertex_embeddings, logger)

    model.to(args.device)

    evaluator = Evaluator(config, dataset, logger)

    log_dir = config["log_dir"]

    max_f1 = -1
    best_task_f1 = {}

    while True:
        models = glob.glob(f'{log_dir}/model__*')
        emb_models = glob.glob(f'{log_dir}/vertex_*')

        def extract_model_key(x):
            return int(re.findall(r'\d+', x)[0])

        models = sorted(models, key=extract_model_key)
        emb_models = sorted(emb_models, key=extract_model_key)

        for i, model_name in enumerate(models):
            if "max" in model_name:
                continue
            tmp_checkpoint_path = os.path.join(log_dir, "model.tmp")
            tmp_emb_path = os.path.join(log_dir, "vertex.tmp")
            shutil.move(model_name, tmp_checkpoint_path)
            shutil.move(f"{emb_models[i]}", tmp_emb_path)
            model.load_state_dict(torch.load(tmp_checkpoint_path))
            vertex_cpt = torch.load(tmp_emb_path)
            vertex_embeddings.load_state_dict(vertex_cpt)

            eval_summary, f1, task_to_f1 = evaluate_for_mode(model, dataset, evaluator)

            if f1 > max_f1:
                max_f1 = f1


                for task, f1 in task_to_f1.items():
                    best_task_f1[task] = f1

                shutil.copy(tmp_checkpoint_path, f"{log_dir}/model.max")
                shutil.copy(tmp_emb_path, f"{log_dir}/vertex.max)")

            logger.info(f"Current max combined F1: {max_f1}")

            for task, f1 in best_task_f1.items():
                logger.info(f"Max {task} F1: {f1}")

            os.remove(tmp_checkpoint_path)
            os.remove(tmp_emb_path)

        time.sleep(config["eval_sleep_secs"])

def evaluate_for_mode(model, dataset, evaluator):
    # Load batch of sentences for each document
    data_iter = data.Iterator(
        dataset.test_dataset,
        1,
        # device=args.device,
        sort_key=lambda x: len(x.text_len),
        repeat=False,
        train=False
    )

    predictions = {}
    total_loss = 0
    for count, batch in enumerate(data_iter):
        with torch.no_grad():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

            doc_batch = dataset.fix_batch(batch)

            predict_dict, loss = model(doc_batch)

            predictions[batch.doc_key[0]] = predict_dict

            total_loss += loss
    evaluator.evaluate(predictions, total_loss)

    return evaluator.summarize_results()


if __name__ == "__main__":
    args = pargs()
    main(args)
