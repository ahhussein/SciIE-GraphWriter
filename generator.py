import torch
from document_dataset import DocumentDataset
from GraphWriter.models.newmodel import model as graph
from GraphWriter.pargs import pargs,dynArgs
import util
import os
from torchtext import data
from models.vertex_embeddings import VertexEmbeddings
from torch import nn


#import utils.eval as evalMetrics

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

def test(args,ds, graph_model,epoch='cmdline'):
  args.vbsz = 1
  model = args.save.split("/")[-1]
  ofn = "outputs/"+model+".beam_predictions"
  #sci_model.eval()
  graph_model.eval()
  k = 0
  test_iter = data.Iterator(
    dataset_wrapper.test_dataset,
    1,
    # device=args.device,
    sort_key=lambda x: len(x.text_len),
    repeat=False,
    train=False
  )
  ofn = "outputs/"+model+".inputs.beam_predictions."+epoch
  pf = open(ofn,'w')
  preds = []
  golds = []

  for b in test_iter:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    with torch.no_grad():
      #if k == 10: break
      print(k,len(test_iter))
      b = ds.fix_batch(b)
      '''
      p,z = m(b)
      p = p[0].max(1)[1]
      gen = ds.reverse(p,b.rawent)
      '''
      # sci_model(b)
      gen = graph_model.beam_generate(b,beamsz=4,k=6)
      gen.sort()
      # TODO pass ents
      gen = ds.reverse(gen.done[0].words,b.rawent)
      k+=1
      # TODO pass ents
      gold = ds.reverse(b.out[0][0][1:],b.rawent)
      print(gold)
      print(gen)
      print()
      preds.append(gen.lower())
      golds.append(gold.lower())
      #tf.write(ent+'\n')
      pf.write(gen.lower()+'\n')

  #sci_model.train()
  graph_model.train()

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

  # TODO read the models dynamically
  # sci_model_name = 'model__4.loss-86.47219597952706.lr-0.0004985014995'
  # sci_model_name = 'model__3.loss-102.81670368739537.lr-0.0004990005'

  graph_model_name = 'graph_model__1.loss-0.0.lr-0.02'
  vertex_model_name = 'vertex_embeddings__1'
  #graph_model_name = 'graph_model__3.loss-102.81670368739537.lr-0.1'

  args = pargs()
  args.eval = True
  config = util.get_config("experiments.conf")[exp_name]
  dataset_wrapper = DocumentDataset(config, args, is_eval=True)
  args = dynArgs(args)


  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], exp_name))

  # Load sci model
  #model = Model(config, dataset_wrapper)
  #sci_cpt = torch.load(f"{config['log_dir']}/{sci_model_name}")
  #model.load_state_dict(sci_cpt)

  # Load graph model
  vertex_embeddings = VertexEmbeddings(config, dataset_wrapper)
  graph_model = graph(args, dataset_wrapper.config, dataset_wrapper, vertex_embeddings)
  graph_cpt = torch.load(f"{config['log_dir']}/{graph_model_name}", map_location='cuda:0')
  graph_model.load_state_dict(graph_cpt)
  m = graph_model.to(args.device)
  vertex_cpt = torch.load(f"{config['log_dir']}/{vertex_model_name}")
  vertex_embeddings.load_state_dict(vertex_cpt)

  graph_model.args = args
  graph_model.maxlen = args.max


  # TODO

  graph_model.starttok = dataset_wrapper.out.vocab.stoi['<start>']
  graph_model.endtok = dataset_wrapper.out.vocab.stoi['<eos>']
  graph_model.eostok = dataset_wrapper.out.vocab.stoi['.']
  args.vbsz = 1
  preds,gold = test(args,dataset_wrapper ,graph_model)
  '''
  scores = metrics(preds,gold)
  for k,v in scores.items():
    print(k+'\t'+str(scores[k]))
  '''
