import torch
from torch import nn
from GraphWriter.models.attention import MultiHeadAttention, MatrixAttn, MultiHeadAttention2
from GraphWriter.models.list_encoder import list_encode, lseq_encode
from GraphWriter.models.last_graph import graph_encode
from GraphWriter.models.beam import Beam
from GraphWriter.models.splan import splanner

class model(nn.Module):
  def __init__(self,args, config):
    super().__init__()
    super().__init__()
    self.args = args
    self.args.ntoks = config.ntoks
    cattimes = 3 if args.title else 2

    self.emb = nn.Embedding(config.ntoks,args.hsz)
    self.lstm = nn.LSTMCell(args.hsz*cattimes,args.hsz)

    self.out = nn.Linear(
      args.hsz*cattimes,
      #args.tgttoks # TODO target tokens
      config.ntoks
    )

    #self.le = list_encode(args)
    self.entout = nn.Linear(args.hsz,1)
    self.switch = nn.Linear(args.hsz*cattimes,1)
    self.attn = MultiHeadAttention2(args.hsz,args.hsz,args.hsz,h=4,dropout_p=args.drop)
    self.mattn = MatrixAttn(args.hsz*cattimes,args.hsz)
    self.graph = (args.model in ['graph','gat','gtrans'])

    if self.graph:
      self.ge = graph_encode(args)
    if args.plan:
      self.splan = splanner(args)

    # TODO
    args.title = False

    if args.title:
      self.tenc = lseq_encode(args,toks=args.ninput)
      self.attn2 = MultiHeadAttention(args.hsz,args.hsz,args.hsz,h=4,dropout_p=args.drop)
      self.mix = nn.Linear(args.hsz,1)

  def forward(self,b):
    if self.args.title:
      # batch of sentences. passed as tensor of plain wordss

      tencs,_ = self.tenc(b.src)

      # b.src[1] sentences length, mask is used to discard padding
      tmask = self.maskFromList(tencs.size(),b.src[1]).unsqueeze(1)
    outp,_ = b.out
    ents = b.top_spans
    entlens = b.doc_num_entities

    if self.graph:
      # rel[0] is adj, rel[1] is rel array
      gents,glob,grels = self.ge(b.adj,b.rels,(b.top_spans,b.doc_num_entities))
      hx = glob
      keys,mask = grels
      # Flip the mask
      mask = mask==0
    else:
      mask = self.maskFromList(ents.size(),entlens)
      hx = ents.mean(dim=1)
      keys = ents
    mask = mask.unsqueeze(1)
    if self.args.plan:
      planlogits = self.splan(hx,keys,mask.clone(),entlens,b.sordertgt)
      schange = (outp==self.args.dottok).t()
      mask.fill_(0)
      planplace = torch.zeros(hx.size(0)).long()
      for i,m in enumerate(b.sorder):
        mask[i][0][m[0]]=1
    else:
      planlogits = None


    # Glob Removes gradient backward path
    # B x hz
    cx = hx.clone().detach().requires_grad_(True)
    a = torch.zeros_like(hx) #self.attn(hx.unsqueeze(1),keys,mask=mask).squeeze(1)

    if self.args.title:
      # Attend last entity of each sample graph to each word in sentence in the title
      # B x hz
      # tencs size B x max_seq_length (rest padded and thats why we use mask) x hz
      a2 = self.attn2(hx.unsqueeze(1),tencs,mask=tmask).squeeze(1)

      #B * 2hz
      a = torch.cat((a,a2),1)
    #e = outp.transpose(0,1)

    # max max length of words * (B) * hz?
    e = self.emb(outp).transpose(0,1)
    outputs = []
    # each 1,2,3 word in a batch
    for i, k in enumerate(e):
      #k = self.emb(k)
      if self.args.plan:
        if schange[i].nonzero().size(0)>0:
          planplace[schange[i].nonzero().squeeze()]+=1
          for j in schange[i].nonzero().squeeze(1):
            if planplace[j]<len(b.sorder[j]):
              mask[j] = 0
              m = b.sorder[j][planplace[j]]
              mask[j][0][b.sorder[j][planplace[j]]]=1

      # check dimensions
      # Mixing word in position x with output of attention of last graph entity and title
      prev = torch.cat((a,k),1)
      hx,cx = self.lstm(prev,(hx,cx))
      a = self.attn(hx.unsqueeze(1),keys,mask=mask).squeeze(1)
      if self.args.title:
        a2 = self.attn2(hx.unsqueeze(1),tencs,mask=tmask).squeeze(1)
        #a =  a + (self.mix(hx)*a2)
        a = torch.cat((a,a2),1)
      out = torch.cat((hx,a),1)
      outputs.append(out)
    l = torch.stack(outputs,1)

    ## -- Experiment 1: Commenting out all copy and switch related vars and only consider generating voca
    #s = torch.sigmoid(self.switch(l))
    o = self.out(l)
    o = torch.softmax(o,2)
    #o = s*o

    #compute copy attn
    #_, z = self.mattn(l,(ents,entlens))
    #z = torch.softmax(z,2)
    #z = (1-s)*z
    #o = torch.cat((o,z),2)
    o = o+(1e-6*torch.ones_like(o))
    return o.log(),planlogits

  def maskFromList(self,size,l):
    # size[1] = max sample (e.g. sentence) length
    # size[0] = batch size
    mask = torch.arange(0,size[1]).unsqueeze(0).repeat(size[0],1).long().cuda()
    mask = (mask <= l.unsqueeze(1))
    mask = mask==0
    return mask
    
  def emb_w_vertex(self,outp,vertex):
    mask = outp>=self.args.ntoks
    if mask.sum()>0:
      idxs = (outp-self.args.ntoks)
      idxs = idxs[mask]
      verts = vertex.index_select(1,idxs)
      outp.masked_scatter_(mask,verts)

    return outp

  def beam_generate(self,b,beamsz,k):
    if self.args.title:
      tencs,_ = self.tenc(b.src)
      tmask = self.maskFromList(tencs.size(),b.src[1]).unsqueeze(1)
    ents = b.top_spans
    entlens = b.doc_num_entities

    if self.graph:
      gents,glob,grels = self.ge(b.adj,b.rels,(b.top_spans,b.doc_num_entities))

      hx = glob
      #hx = ents.max(dim=1)[0]
      keys,mask = grels
      mask = mask==0
    else:
      mask = self.maskFromList(ents.size(),entlens)
      hx = ents.max(dim=1)[0]
      keys =ents
    mask = mask.unsqueeze(1)
    if self.args.plan:
      planlogits = self.splan.plan_decode(hx,keys,mask.clone(),entlens)
      print(planlogits.size())
      sorder = ' '.join([str(x) for x in planlogits.max(1)[1][0].tolist()])
      print(sorder)
      sorder = [x.strip() for x in sorder.split("-1")]
      sorder = [[int(y) for y in x.strip().split(" ")] for x in sorder]
      mask.fill_(0)
      planplace = torch.zeros(hx.size(0)).long()
      for i,m in enumerate(sorder):
        mask[i][0][m[0]]=1
    else:
      planlogits = None

    cx = hx.clone().detach().requires_grad_(True)
    a = self.attn(hx.unsqueeze(1),keys,mask=mask).squeeze(1)

    if self.args.title:
      a2 = self.attn2(hx.unsqueeze(1),tencs,mask=tmask).squeeze(1)
      a = torch.cat((a,a2),1)
    outputs = []

    # TODO ensure that this represents the number of documents
    outp = torch.LongTensor(ents.size(0),1).fill_(self.starttok)
    beam = None
    for i in range(self.maxlen):
      # TODO nerd
      op = self.emb_w_vertex(outp.clone(),None)
      if self.args.plan:
        schange = op==self.args.dottok
        if schange.nonzero().size(0)>0:
          print(schange, planplace, sorder)
          planplace[schange.nonzero().squeeze()]+=1
          for j in schange.nonzero().squeeze(1):
            if planplace[j]<len(sorder[j]):
              mask[j] = 0
              m = sorder[j][planplace[j]]
              mask[j][0][sorder[j][planplace[j]]]=1
      op = self.emb(op).squeeze(1)
      prev = torch.cat((a,op),1)
      hx,cx = self.lstm(prev,(hx,cx))
      a = self.attn(hx.unsqueeze(1),keys,mask=mask).squeeze(1)
      if self.args.title:
        a2 = self.attn2(hx.unsqueeze(1),tencs,mask=tmask).squeeze(1)
        #a =  a + (self.mix(hx)*a2)
        a = torch.cat((a,a2),1)
      l = torch.cat((hx,a),1).unsqueeze(1)
      #s = torch.sigmoid(self.switch(l))
      o = self.out(l)
      o = torch.softmax(o,2)

      # TODO copy
      #o = s*o
      #compute copy attn
      #_, z = self.mattn(l,(ents,entlens))
      #z = torch.softmax(z,2)
      #z = (1-s)*z
      #o = torch.cat((o,z),2)
      o[:,:,0].fill_(0)
      o[:,:,1].fill_(0)
      '''
      if beam:
        for p,q in enumerate(beam.getPrevEnt()):
          o[p,:,q].fill_(0)
        for p,q in beam.getIsStart():
          for r in q:
            o[p,:,r].fill_(0)
      '''

      o = o+(1e-6*torch.ones_like(o))
      decoded = o.log()
      scores, words = decoded.topk(dim=2,k=k)
      if not beam:
        beam = Beam(words.squeeze(),scores.squeeze(),[hx for i in range(beamsz)],
                  [cx for i in range(beamsz)],[a for i in range(beamsz)],beamsz,k,self.args.ntoks)
        beam.endtok = self.endtok
        beam.eostok = self.eostok
        keys = keys.repeat(len(beam.beam),1,1)
        mask = mask.repeat(len(beam.beam),1,1)
        if self.args.title:
          tencs = tencs.repeat(len(beam.beam),1,1)
          tmask = tmask.repeat(len(beam.beam),1,1)
        if self.args.plan:
          planplace= planplace.unsqueeze(0).repeat(len(beam.beam),1)
          sorder = sorder*len(beam.beam)

          
        ents = ents.repeat(len(beam.beam),1,1)
        entlens = entlens.repeat(len(beam.beam))
      else:
        if not beam.update(scores,words,hx,cx,a):
          break
        keys = keys[:len(beam.beam)]
        mask = mask[:len(beam.beam)]
        if self.args.title:
          tencs = tencs[:len(beam.beam)]
          tmask = tmask[:len(beam.beam)]
        if self.args.plan:
          planplace= planplace[:len(beam.beam)]
          sorder = sorder[0]*len(beam.beam)
        ents = ents[:len(beam.beam)]
        entlens = entlens[:len(beam.beam)]
      outp = beam.getwords()
      hx = beam.geth()
      cx = beam.getc()
      a = beam.getlast()

    return beam
