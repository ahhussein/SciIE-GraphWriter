import pyhocon
import os
import errno

def get_config(filename):
  return pyhocon.ConfigFactory.parse_file(filename)

def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path

def print_config(config):
  print(pyhocon.HOCONConverter.convert(config, "hocon"))

def set_gpus(*gpus):
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
  print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

def flatten(l):
  return [item for sublist in l for item in sublist]

def print_ner(sent_example):
  for ner in sent_example['ner']:
    print_range(
      sent_example['sentence'],
      ner[0] - sent_example['word_offset'],
      ner[1] - sent_example['word_offset']
    )
    print(f" => {ner[2]}")

def print_relations(sent_example):
  for rel in sent_example['relations']:
    # Print first entity
    print_range(
      sent_example['sentence'],
      rel[0] - sent_example['word_offset'],
      rel[1] - sent_example['word_offset']
    )

    print(" == ", end='')

    # Print second entity
    print_range(
      sent_example['sentence'],
      rel[2] - sent_example['word_offset'],
      rel[3] - sent_example['word_offset']
    )

    print(" => ", end='')

    # Print relation type
    print(rel[4])

def print_range(text, i, j):
  while i <= j:
    print(f"{text[i]} ", end='')
    i = i + 1


def printexit(x):
  print(x)
  exit()