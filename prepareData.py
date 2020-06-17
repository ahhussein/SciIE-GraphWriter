import json

"""
Prepare data to include title and splits into train/val/test
"""
train = 0.4
val = 0.3
with open('corpus-export.json') as f:
  exportList = []
  for line in f:
    data = json.loads(line)
    export = data['abstract']
    export['doc_key'] = data['doc_key']
    export['title'] = ' '.join([' '.join(x) for x in data['title']['sentences']])
    export['clusters'] = export['coref']
    del export['coref']
    exportList.append(export)


trainSlice = int(len(exportList)*train)
valSlice = int(len(exportList) * val)

with open('train.json', 'a') as outfile:
  for line in exportList[:trainSlice]:
    outfile.write(json.dumps(line)+'\n')

with open('dev.json', 'w') as outfile:
  for line in exportList[trainSlice:trainSlice+valSlice]:
    outfile.write(json.dumps(line)+'\n')

with open('test.json', 'w') as outfile:
  for line in exportList[trainSlice+valSlice:]:
    outfile.write(json.dumps(line)+'\n')



