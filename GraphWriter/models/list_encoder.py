import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from allennlp.modules.elmo import Elmo

class lseq_encode(nn.Module):

  def __init__(self,args,vocab=None,toks=None):
    super().__init__()
    if vocab:
      self.elmo = Elmo(args.options_file, args.weight_file, 1, dropout=0.5,vocab_to_cache=vocab)
      toks = len(vocab)
      sz = args.esz+512
      self.use_elmo = True
    else:
      self.use_elmo = False
      sz = args.esz
    self.lemb = nn.Embedding(toks,args.esz)
    nn.init.xavier_normal_(self.lemb.weight)
    self.input_drop = nn.Dropout(args.embdrop)
      
    self.encoder = nn.LSTM(sz,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)

  def _cat_directions(self, h):
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

  def forward(self,inp):
    # Batch of titles at a time. Each batch is a list of tuples
    # tuple1 contains the sentences encoded and the tuple2 contains the lengths
    l, ilens = inp
    learned_emb = self.lemb(l)
    learned_emb = self.input_drop(learned_emb)
    if self.use_elmo:
      elmo_emb = self.elmo(l,word_inputs=l)
      e = torch.cat((elmo_emb['elmo_representations'][0],learned_emb),2)
    else:
      e = learned_emb

    # sent_len = sorted length, idxs= original idxs before sorting
    sent_lens, idxs = ilens.sort(descending=True)
    # rearrange r to match idxs, longest seq first
    e = e.index_select(0,idxs)

    # padding was necessary to train batch embeddings, now it's necessary to pack to avoid high computation
    e = pack_padded_sequence(e,sent_lens,batch_first=True)

    # output is packed, e is the output of (seq_len, batch, hidden_size)
    # h is the otput of the last time seq (hidden_layer, batch)
    # c is the cell state
    e, (h,c) = self.encoder(e)
    # 0 to discard lengths
    e = pad_packed_sequence(e,batch_first=True)[0]
    e = torch.zeros_like(e).scatter(0,idxs.unsqueeze(1).unsqueeze(1).expand(-1,e.size(1),e.size(2)),e)
    h = h.transpose(0,1)
    h = torch.zeros_like(h).scatter(0,idxs.unsqueeze(1).unsqueeze(1).expand(-1,h.size(1),h.size(2)),h)
    return e,h

class list_encode(nn.Module):
  def __init__(self,args):
    super().__init__()
    # sequence encoder
    self.seqenc = lseq_encode(args,toks=args.vtoks)#,vocab=args.ent_vocab)
    #self.seqenc = lseq_encode(args,vocab=args.ent_vocab)

  def pad(self,tensor,length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

  def forward(self,batch,pad=True):
    batch,phlens,batch_lens = batch
    batch_lens = tuple(batch_lens.tolist())

    # Batch is equal to 32 * number of sentences or sum of all entities in the 32 batch * maxlen sentence

    _,enc = self.seqenc((batch,phlens))
    # discard first layer output
    enc = enc[:,2:]

    # cat two directions
    # get (entity * hidden_size)
    enc = torch.cat([enc[:,i] for i in range(enc.size(1))],1)

    # Max # entity per sample.
    m = max(batch_lens)

    # list of all entities matrics padded to the max entity length
    encs = [self.pad(x,m) for x in enc.split(batch_lens)]

    # Stack them to end up with 32 * maxlen_of_entities * hidden size
    out = torch.stack(encs,0)
    return out

