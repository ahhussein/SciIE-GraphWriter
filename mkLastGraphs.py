import sys
import json
from collections import defaultdict
outf = open("preRelease.tsv",'w') 
relvocab = []
with open('corpus.json') as f:
  for line in f:
    data = json.loads(line)
    words = [x.strip() for y in data['abstract']['sentences'] for x in y]
    if len([x for x in words if len(x)<2]) > (0.25 * len(words)):
      continue
    title = ' '.join([' '.join(x) for x in data['title']['sentences']])
    # Abstract
    text = ' '.join([' '.join(x) for x in data['abstract']['sentences']])
    textidx = text.split(" ")
    corefd = {}
    corefbak = {}
    reltos = {}
    for chain in data['abstract']['coref']:
      mentions = [' '.join(textidx[x[0]:x[1]+1]).strip() for x in chain]
      mentions = [x for x in mentions if len(x)>0]
      # Sort longest mention first
      mentions.sort(key=lambda x:len(x.split(" ")),reverse=True)

      cannon = mentions[0]
      for m in mentions:
        if m not in corefd:
          corefd[m] = cannon
        else:
          if len(corefd[m].split(" "))<len(cannon.split(" ")):
            corefd[m] = cannon
      corefbak[cannon] = mentions

    todo = True
    while todo:
      todo=False
      for k,v in corefd.items():
        if v in corefd and corefd[v] != v:
          corefd[k] = corefd[v]
          todo = True

    # Entities longest mention, unique
    entsort = []#[v for k,v in corefd.items()]
    entsbak = [(v,"") for k,v in corefd.items()]

    # dict of entity longest mention -> label
    nerd = {}
    for i,s in enumerate(data['abstract']['sentences']):
      nernums = data['abstract']['ner'][i]
      for x in nernums:
        a = ' '.join(s[x[0]:x[1]+1])
        entsbak.append((a,x[2]))

        # Find entity longest mention
        if a in corefd:
          a = corefd[a]

        if a not in nerd:
          nerd[a] = x[2]
        entsort.append(a)
    entsort = [corefd[x] if x in corefd else x for x in entsort]
    entsort = [x for x in entsort if nerd[x] in ["Method","Task","Material","Metric","OtherScientificTerm"]]
    entsort = list(set(entsort))
    entsort.sort(key=lambda x: len(x.split(" ")),reverse=True)
    if not entsort:
      continue


    '''
    for m,c in corefd.items():
      m = " "+m+" "
      c = " "+c+" "
      text = text.replace(m,c)
    '''
    entorder = []
    entsbak = [k for k,v in entsbak]
    entsbak.sort(key=lambda x: len(x.split(" ")),reverse=True)
    for i,e in enumerate(entsbak):
      ent = "ENTITY"
      if e in nerd:
        ent = nerd[e]
      if e in corefd:
        canne = corefd[e]
      else:
        canne = e
      if canne not in entsort:
        continue
      entorder.append(canne)
      e = " "+e+" "
      canne = canne.replace(" ","_")
      text = text.replace(e, "#"+canne+"#")
    ents = list(set(entorder))
    ents.sort(key=lambda x: len(x.split(" ")),reverse=True)
    for i,x in enumerate(ents):
      n = nerd[x]
      x2 = " <"+n+"_"+str(i)+"> "

      e = "#"+x.replace(" ","_")+"#"

      text =  text.replace(e, x2)

    allrels = []
    for i,s in enumerate(data['abstract']['sentences']):
      relnums = data['abstract']['relation'][i]
      for x in relnums:
        a = ' '.join(s[x[0]:x[1]+1])
        b = ' '.join(s[x[2]:x[3]+1])
        if a in corefd:
          a = corefd[a]
        if b in corefd:
          b = corefd[b]
        if a not in ents or b not in ents:
          continue
        a = str(ents.index(a))
        b = str(ents.index(b))

        if x[4] not in relvocab:
          relvocab.append(x[4])
        allrels.append(' '.join([a,str(relvocab.index(x[4])),b]))
        reltos[allrels[-1]]=i
    allrels = list(set(allrels))
    if not allrels:
      continue

    entsentorder = []
    for i,x in enumerate(ents):
      n = nerd[x]
      x2 = " <"+n+"_"+str(i)+"> "
      scount =text.split(x2)[0].count(" . ")
      wcount = len(text.split(x2)[0].split(" "))
      entsentorder.append((scount*100+wcount,x))
    entsentorder.sort()
    relorder = []
    entsandrels = ents+["ROOOT"]+allrels
    for i in range(len(data['abstract']['sentences'])):
      senti = [x[1] for x in entsentorder if x[0]//100 == i]+[x for x in allrels if reltos[x]==i]
      idxs = [entsandrels.index(x) for x in senti]
      relorder.extend(idxs)
      relorder.append(entsandrels.index("ROOOT"))
      relorder.append(-1)


    text = text.strip().split(" ")
    text = ' '.join(text).lower()

    entner = ' '.join(["<"+nerd[x]+">" for i,x in enumerate(ents)]).lower()
    for i in range(len(ents)):
      k = ents[i].split(" ")
      if '-LRB-' in k and '-RRB-' in k:
        a = k.index('-LRB-')
        b = k.index('-RRB-')
        if b-a == 2 and a>0:
          print(ents[i])
          k = k[:a]+k[b+1:]
        ents[i] = ' '.join(k)
    entstr = ' ; '.join([x for i,x in enumerate(ents)]).lower()
    relstr = ' ; '.join(allrels)
    relorder = " ".join([str(i) for i in relorder])
    outstr = title + '\t' + entstr + '\t' + entner + "\t" + relstr + '\t' + text + "\n"#"\t" + relorder+"\n"
    #print(outstr);exit()
    outf.write(outstr)
outf.close()
with open("relations.vocab",'w') as f:
  f.write('\n'.join(relvocab))



