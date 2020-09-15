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


torch.manual_seed(0)

logger = logging.getLogger('myapp')
hdlr = logging.FileHandler('predict.log')
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

    model_eval_name = 'model__3.loss-0.0.lr-0.0004990005'

    vertex_embeddings = VertexEmbeddings(config, dataset)

    model = Model(config, dataset, vertex_embeddings, logger)

    model.to(args.device)

    evaluator = Evaluator(config, dataset, logger)

    log_dir = config["log_dir"]

    model.load_state_dict(torch.load(f"{log_dir}/{model_eval_name}"))

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
    model.eval()
    for count, batch in enumerate(data_iter):
        with torch.no_grad():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

            doc_batch = dataset.fix_batch(batch)
            predict_dict, loss = model(doc_batch)

            predictions[batch.doc_key[0]] = predict_dict

            logger.info("Evaluated {}/{} documents.".format(count + 1, len(evaluator.coref_eval_data)))

            total_loss += loss
    evaluator.evaluate(predictions, total_loss)
    evaluator.write_out()


if __name__ == "__main__":
    args = pargs()
    main(args)
