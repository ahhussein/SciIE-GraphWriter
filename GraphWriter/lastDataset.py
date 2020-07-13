import torch
from collections import Counter
import dill
from torchtext import data
import pargs as arg
from copy import copy

class dataset:

  def __init__(self, args):
    args.path = args.datadir + args.data
    print("Loading Data from ",args.path)
    self.args = args
    self.mkVocabs(args)
    print("Vocab sizes:")
    for x in self.fields:
      try:
        print(x[0],len(x[1].vocab))
      except:
        try:
          print(x[0],len(x[1].itos))
        except:
          pass

  def build_ent_vocab(self,path,unkat=0):
    ents = ""
    with open(path) as f:
      for l in f:
        ents +=  " "+l.split("\t")[1]

    itos = sorted(list(set(ents.split(" "))))
    itos[0] == "<unk>"; itos[1] == "<pad>"
    stoi = {x:i for i,x in enumerate(itos)}
    return itos,stoi

  def vec_ents(self,ex,field):
    # returns tensor and lens
    ex = [[field.stoi[x] if x in field.stoi else 0 for x in y.strip().split(" ")] for y in ex.split(";")]
    return self.pad_list(ex,1)
  
  def mkGraphs(self,r,ent):
    # ent represents how many entities
    #convert triples to entlist with adj and rel matrices
    pieces = r.strip().split(';')
    # Array of relations [[1,2,3], [2,3,4]]
    x = [[int(y) for y in z.strip().split()] for z in pieces]
    # ROOT / Global NODE
    rel = [2]
    #global root node + 2 nodes per each relation + number oe entities in that paper abstract
    adjsize = ent+1+(2*len(x))
    adj = torch.zeros(adjsize,adjsize)

    # global node
    for i in range(ent):
      adj[i,ent]=1
      adj[ent,i]=1

    # Self connection
    for i in range(adjsize):
      adj[i,i]=1

    # converting relations into nodes
    for y in x:

      # skip special chars, inverse
      rel.extend([y[1]+3,y[1]+3+self.REL.size])
      a = y[0]
      b = y[2]
      c = ent+len(rel)-2
      d = ent+len(rel)-1
      adj[a,c] = 1 
      adj[c,b] = 1
      adj[b,d] = 1 
      adj[d,a] = 1
    rel = torch.LongTensor(rel)
    return (adj,rel)
  
  def adjToSparse(self,adj):
    sp = []
    for row in adj:
      sp.append(row.nonzero().squeeze(1))
    return sp

  def mkVocabs(self,args):
    args.path = args.datadir + args.data
    # title: e.g. Hierarchical Semantic Classification : Word Sense Disambiguation with World Knowledge
    # TODO transformation - create batches on fly
    # TODO transformation - build vocab beforehand
    self.INP = data.Field(sequential=True, batch_first=True,init_token="<start>", eos_token="<eos>",include_lengths=True)

    # Processed Abstract: e.g. we present a <method_6> for <task_1> that supplements <material_4> with <material_5> encoding general '' world knowledge '' . the <method_6> compiles knowledge contained in a dictionary-ontology into additional training data ,
    # and integrates <material_0> through a novel <method_3> . experiments on a <task_2> provide empirical evidence that this '' <method_3> '' outperforms a state-of-the-art standard '' flat '' one .
    # TODO transformation - build vocab beforehand - Generate all ner types with numbers up to 40
    self.OUTP = data.Field(sequential=True, batch_first=True,init_token="<start>", eos_token="<eos>",include_lengths=True)

    # Target text after (final output)
    self.TGT = data.Field(sequential=True, batch_first=True,init_token="<start>", eos_token="<eos>")

    # extracted entity types: e.g. <material> <task> <task> <method> <material> <material> <method>
    # TODO transformation - build vocab beforehand
    self.NERD = data.Field(sequential=True, batch_first=True,eos_token="<eos>")

    # extracted entities (Graph) from abstract: e.g. task-specific and background data ; lexical semantic classification problems ; word sense disambiguation task ; hierarchical learning architecture ; task-specific training data ; background data ; learning architecture
    self.ENT = data.RawField()

    # relation between entities (0 based): e.g. 6 0 1
    self.REL = data.RawField()

    # token order: e.g. 6 1 4 5 8 7 -1 0 3 7 -1 2 7 -1 (not yet clear)
    self.SORDER = data.RawField()
    self.SORDER.is_target = False
    self.REL.is_target = False 
    self.ENT.is_target = False

    self.fields=[("src",self.INP),("ent",self.ENT),("nerd",self.NERD),("rel",self.REL),("out",self.OUTP),("sorder",self.SORDER)]
    train = data.TabularDataset(path=args.path, format='tsv',fields=self.fields)
    print('building vocab')

    self.OUTP.build_vocab(train, min_freq=args.outunk)
    # Exxtend the outpt vocab to contain these tokens (They exist in the original vocab but with numbers)
    generics =['<method>','<material>','<otherscientificterm>','<metric>','<task>']
    self.OUTP.vocab.itos.extend(generics)
    for x in generics:
      self.OUTP.vocab.stoi[x] = self.OUTP.vocab.itos.index(x)

    # same copy goes to target
    self.TGT.vocab = copy(self.OUTP.vocab)

    # Extend target to include all the entity typs with numbers up to 40
    specials = "method material otherscientificterm metric task".split(" ")
    for x in specials:
      for y in range(40):
        s = "<"+x+"_"+str(y)+">"
        # What about the other way around? # TODO check lengths of boths lists and check for the specific special words
        self.TGT.vocab.stoi[s] = len(self.TGT.vocab.itos)+y
    self.NERD.build_vocab(train,min_freq=0)
    for x in generics:
      self.NERD.vocab.stoi[x] = self.OUTP.vocab.stoi[x]
    self.INP.build_vocab(train, min_freq=args.entunk)

    self.REL.special = ['<pad>','<unk>','ROOT']
    with open(args.datadir+"/"+args.relvocab) as f:
      rvocab = [x.strip() for x in f.readlines()]
      self.REL.size = len(rvocab)
      rvocab += [x+"_inv" for x in rvocab]
      relvocab = self.REL.special + rvocab
    self.REL.itos = relvocab

    # TODO transformation - build vocabs beforehand
    self.ENT.itos,self.ENT.stoi = self.build_ent_vocab(args.path)


    print('done')
    if not self.args.eval:
      self.mkiters(train)

  def listTo(self,l):
    return [x.to(self.args.device) for x in l]

  def fixBatch(self,b):
    # unzip (*)  so that we get ent = (tensor1, tensor2, tensor3) - phlens = (lens1[array], lens2, len3)
    # tensor1 2d, rows sentences, columns for words
    ent,phlens = zip(*b.ent)

    # Pad all sample in a batch with 1 to match the maximum length of words and to unify dimensions
    #returns array of tensors and array of lengths
    ent,elens = self.adjToBatch(ent)

    # Ent is all samples padded in one big matrix
    ent = ent.to(self.args.device)
    adj,rel = zip(*b.rel)

    # TODO read later how to preprocess graph
    if self.args.sparse:
      b.rel = [adj,self.listTo(rel)]
    else:
      b.rel = [self.listTo(adj),self.listTo(rel)]
    if self.args.plan:
      b.sordertgt = self.listTo(self.pad_list(b.sordertgt))

    # Phlens indicates how many actual words in entity
    # (since the entity is padded according to the batch max entity size)
    # one big array of lengths, each record represents actual number of words per entity
    phlens = torch.cat(phlens,0).to(self.args.device)

    # elens is how many entities in one sample since all samples are cat together as rows
    elens = elens.to(self.args.device)
    b.ent = (ent,phlens,elens)
    return b


  def adjToBatch(self,adj):
    # word in entity, # entities in sample, samples in batch

    # adjs is tuple of 2d matrices, rows sentences (words is one field), columns for words
    lens = [x.size(0) for x in adj] # each record is the # of words in a sequence (sample)
    m = max([x.size(1) for x in adj]) #  max of columns for a word for the whole batch

    # PAdding over batch, before was just padding over sentences in the sample
    data = [self.pad(x.transpose(0,1),m).transpose(0,1) for x in adj]

    # Append all tensors together. That's where length tensor will become handy
    # [
    #   # lens[0] = 2 entities in one sample
    #   [120,12123, 123123, 1,  1],
    #   [123, 123, 23123, 123123 ,1],
    #
    #   # lens[1] = 3 words in second entity
    #   [120,12123, 123123, 1,  1],
    #   [120,12123, 123123, 1,  1],
    #   [120, 12123, 123123, 1, 1],
    # ]
    data = torch.cat(data,0)
    return data,torch.LongTensor(lens)

  def bszFn(self,e,l,c):
    return c+len(e.out)

  # TODO transformation per batch
  def mkiters(self,train):
    args = self.args
    c = Counter([len(x.out) for x in train])
    t1,t2,t3 = [],[],[]
    print("Sorting training data by len")
    for x in train:
      l = len(x.out)
      if l<100:
        t1.append(x)
      elif l>100 and l<220:
        t2.append(x)
      else:
        t3.append(x)
    t1d = data.Dataset(t1,self.fields)
    t2d = data.Dataset(t2,self.fields)
    t3d = data.Dataset(t3,self.fields)
    valid = data.TabularDataset(path=args.path.replace("train","val"), format='tsv',fields=self.fields)
    print("ds sizes:",end='\t')
    for ds in [t1d,t2d,t3d,valid]:
      print(len(ds.examples),end='\t')
      for x in ds:
        x.rawent = x.ent.split(" ; ")
        # seperate one record entries by ; and for each record, a vector is build for each word by splitting with a space [[1,2], [123,123123]]
        x.ent = self.vec_ents(x.ent,self.ENT)

        # small graph for each record. One graph for each paper
        x.rel = self.mkGraphs(x.rel,len(x.ent[1]))
        if args.sparse:
          x.rel = (self.adjToSparse(x.rel[0]),x.rel[1])

        # out = target
        x.tgt = x.out

        # remove the entity number from entity type
        x.out = [y.split("_")[0]+">" if "_" in y else y for y in x.out]

        x.sordertgt = torch.LongTensor([int(y)+3 for y in x.sorder.split(" ")])
        x.sorder = [[int(z) for z in y.strip().split(" ")] for y in x.sorder.split("-1")[:-1]]

      ds.fields["tgt"] = self.TGT
      ds.fields["rawent"] = data.RawField()
      ds.fields["sordertgt"] = data.RawField()

    self.t2_iter = data.Iterator(t2d,args.t2size,device=args.device,sort_key=lambda x:len(x.out),repeat=False,train=True)
    self.t3_iter = data.Iterator(t3d,args.t3size,device=args.device,sort_key=lambda x:len(x.out),repeat=False,train=True)
    self.val_iter= data.Iterator(valid,args.t3size,device=args.device,sort_key=lambda x:len(x.out),sort=False,repeat=False,train=False)

  def mktestset(self, args):
    path = args.path.replace("train",'test')
    fields=self.fields
    ds = data.TabularDataset(path=path, format='tsv',fields=fields)
    ds.fields["rawent"] = data.RawField()
    for x in ds:
      x.rawent = x.ent.split(" ; ")
      x.ent = self.vec_ents(x.ent,self.ENT)
      x.rel = self.mkGraphs(x.rel,len(x.ent[1]))
      if args.sparse:
        x.rel = (self.adjToSparse(x.rel[0]),x.rel[1])
      x.tgt = x.out
      x.out = [y.split("_")[0]+">" if "_" in y else y for y in x.out]
      x.sordertgt = torch.LongTensor([int(y)+3 for y in x.sorder.split(" ")])
      x.sorder = [[int(z) for z in y.strip().split(" ")] for y in x.sorder.split("-1")[:-1]]
    ds.fields["tgt"] = self.TGT
    ds.fields["rawent"] = data.RawField()
    ds.fields["sordertgt"] = data.RawField()
    dat_iter = data.Iterator(ds,1,device=args.device,sort_key=lambda x:len(x.src), train=False, sort=False)
    return dat_iter

  def rev_ents(self,batch):
    vocab = self.NERD.vocab
    es = []
    for e in batch:
      s = [vocab.itos[y].split(">")[0]+"_"+str(i)+">" for i,y in enumerate(e) if vocab.itos[y] not in ['<pad>','<eos>']]
      es.append(s)
    return es

  def reverse(self,x,ents):
    ents = ents[0]
    vocab = self.TGT.vocab
    s = ' '.join([vocab.itos[y] if y<len(vocab.itos) else ents[y-len(vocab.itos)].upper() for j,y in enumerate(x)])   
    #s = ' '.join([vocab.itos[y] if y<len(vocab.itos) else ents[y-len(vocab.itos)] for j,y in enumerate(x)])   
    if "<eos>" in s: s = s.split("<eos>")[0]
    return s

  def relfix(self,relstrs):
    mat = []
    for x in relstrs:
      pieces = x.strip().split(';')
      x = [[int(y)+len(self.REL.special) for y in z.strip().split()] for z in pieces]
      mat.append(torch.LongTensor(x).cuda())
    lens = [x.size(0) for x in mat]
    m = max(lens)
    mat = [self.pad(x,m) for x in mat]
    mat = torch.stack(mat,0)
    lens = torch.LongTensor(lens).cuda()
    return mat,lens

  def getEnts(self,entseq):
    newents = []
    lens = []
    for i,l in enumerate(entseq):
      l = l.tolist()
      if self.enteos in l:
        l = l[:l.index(self.enteos)]
      tmp = []
      while self.entspl in l:
        tmp.append(l[:l.index(self.entspl)])
        l = l[l.index(self.entspl)+1:]
      if l:
        tmp.append(l)
      lens.append(len(tmp))
      tmplen = [len(x) for x in tmp]
      m = max(tmplen)
      tmp = [x +([1]*(m-len(x))) for x in tmp]
      newents.append((torch.LongTensor(tmp).cuda(),torch.LongTensor(tmplen).cuda()))
    return newents,torch.LongTensor(lens).cuda()

  def listToBatch(self,inp):
    data, lens = zip(*inp)
    lens = torch.tensor(lens)
    m = torch.max(lens).item()
    data = [self.pad(x.transpose(0,1),m).transpose(0,1) for x in data]
    data = torch.cat(data,0)
    return data,lens


    

  def rev_rel(self,ebatch,rbatch):
    vocab = self.ENT.vocab
    for i,ents in enumerate(ebatch):
      es = []
      for e in ents:
        s = ' '.join([vocab.itos[y] for y in e])   
        es.append(s)
      rels = rbatch[i]
      for a,r,b in rels:  
        print(es[a],self.REL.itos[r],es[b])
      print()

  def pad_list(self,l,ent=1):  
    lens = [len(x) for x in l]
    m = max(lens)
    # Concatenates sequence of tensors along a new dimension.
    return torch.stack([self.pad(torch.tensor(x),m,ent) for x in l],0), torch.LongTensor(lens)

  def pad(self,tensor, length,ent=1):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(ent)])

  def seqentmat(self,entseq):
    newents = []
    lens = []
    sms = []
    for l in entseq:  
      l = l.tolist()
      if self.enteos in l:
        l = l[:l.index(self.enteos)]
      tmp = []
      while self.entspl in l:
        tmp.append(l[:l.index(self.entspl)])
        l = l[l.index(self.entspl)+1:]
      if l:
        tmp.append(l)
      lens.append(len(tmp))
      m = max([len(x) for x in tmp])
      sms.append(m)
      tmp = [x +([0]*(m-len(x))) for x in tmp]
      newents.append(tmp)
    sm = max(lens)
    pm = max(sms)
    for i in range(len(newents)):
      tmp = torch.LongTensor(newents[i]).transpose(0,1)
      tmp = self.pad(tmp,pm,ent=0)
      tmp = tmp.transpose(0,1)
      tmp = self.pad(tmp,sm,ent=0)
      newents[i] = tmp
    newents = torch.stack(newents,0).cuda()
    lens = torch.LongTensor(lens).cuda()
    return newents,lens

if __name__=="__main__":
  args = arg.pargs()  
  ds = dataset(args)
  ds.getBatch()
  
