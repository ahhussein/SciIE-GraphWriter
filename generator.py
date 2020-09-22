import torch
from document_dataset import DocumentDataset
from GraphWriter.models.newmodel import model as graph
from GraphWriter.pargs import pargs,dynArgs
import util
import os
from torchtext import data
from models.vertex_embeddings import VertexEmbeddings
from torch import nn
import glob, sys
from eval import Evaluate
import logging
import ntpath
#import utils.eval as evalMetrics

logger = logging.getLogger('myapp')
hdlr = logging.FileHandler('eval-graph.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

evaluator = Evaluate()


def tgtreverse(tgts,entlist,order):
  entlist = entlist[0]
  order = [int(x) for x in order[0].split(" ")]
  tgts = tgts.split(" ")
  k = 0
  for i,x in enumerate(tgts):
    if x[0] == "<" and x[-1]=='>':
      tgts[i] = entlist[order[k]]
      k+=1
  return " ".join(tgts)

def test(ds, graph_model, model_name, epoch='cmdline'):
  global evaluator
  k = 0
  test_iter = data.Iterator(
    dataset_wrapper.test_dataset,
    1,
    # device=args.device,
    sort_key=lambda x: len(x.text_len),
    repeat=False,
    train=False
  )
  ofn = "outputs/"+model_name+".inputs.beam_predictions."+epoch
  ofngt = "outputs/"+model_name+".inputs.beam_gt."+epoch
  pf = open(ofn,'w')
  pfgt = open(ofngt,'w')
  preds = []
  golds = []

  for b in test_iter:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    with torch.no_grad():
      b = ds.fix_batch(b)
      '''
      p,z = m(b)
      p = p[0].max(1)[1]
      gen = ds.reverse(p,b.rawent)
      '''
      # sci_model(b)
      gen = graph_model.beam_generate(b,beamsz=4,k=6)
      gen.sort()
      gen = ds.reverse(gen.done[0].words,b.rawent)
      k+=1
      gold = ds.reverse(b.tgt[0][1:],b.rawent)
      preds.append(gen.lower())
      golds.append(gold.lower())
      #tf.write(ent+'\n')
      pf.write(gen.lower()+'\n')
      pfgt.write(gold.lower()+'\n')
  pf.close()
  pfgt.close()
  # get and report evaluation mertices
  with open(ofn) as f:
    cands = {'generated_description' + str(i): x.strip() for i, x in enumerate(f.readlines())}
  with open(ofngt) as f:
    refs = {'generated_description' + str(i): [x.strip()] for i, x in enumerate(f.readlines())}
  final_scores = evaluator.evaluate(live=True, cand=cands, ref=refs)
  logger.info(f"Results for model: {model_name}")
  logger.info(f"Bleu_1:\t {final_scores['Bleu_1']}")
  logger.info(f"Bleu_2:\t {final_scores['Bleu_2']}")
  logger.info(f"Bleu_3:\t {final_scores['Bleu_3']}")
  logger.info(f"Bleu_4:\t {final_scores['Bleu_4']}")
  logger.info(f"ROUGE_L:\t {final_scores['ROUGE_L']}")
  logger.info('METEOR:\t', final_scores['METEOR'])
  #logger.info('CIDEr:\t', final_scores['CIDEr'])

  return preds,golds

'''
def metrics(preds,gold):
  cands = {'generated_description'+str(i):x.strip() for i,x in enumerate(preds)}
  refs = {'generated_description'+str(i):[x.strip()] for i,x in enumerate(gold)}
  x = evalMetrics.Evaluate()
  scores = x.evaluate(live=True, cand=cands, ref=refs)
  return scores
'''

if __name__=="__main__":

  # TODO read dynamically
  exp_name = "scientific_best_ner"

  args = pargs()
  args.eval = True
  config = util.get_config("experiments.conf")[exp_name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], exp_name))
  models = glob.glob(f'{config["log_dir"]}/graph_model*')

  dataset_wrapper = DocumentDataset(config, args, is_eval=True)
  args = dynArgs(args)


  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], exp_name))

  # Load graph model
  vertex_embeddings = VertexEmbeddings(config, dataset_wrapper)
  graph_model = graph(args, dataset_wrapper.config, dataset_wrapper, vertex_embeddings)

  graph_model.args = args
  graph_model.maxlen = args.max
  graph_model.starttok = dataset_wrapper.out.vocab.stoi['<start>']
  graph_model.endtok = dataset_wrapper.out.vocab.stoi['<eos>']
  graph_model.eostok = dataset_wrapper.out.vocab.stoi['.']
  args.vbsz = 1

  m = graph_model.to(args.device)
  models = glob.glob(f'{config["log_dir"]}/graph_model__*')

  graph_model.eval()

  for i, model_name in enumerate(models):
    graph_cpt = torch.load(f"{model_name}", map_location='cuda:0')
    graph_model.load_state_dict(graph_cpt)

    preds, gold = test(dataset_wrapper ,graph_model, ntpath.basename(model_name))

  graph_model.train()

