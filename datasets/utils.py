import pandas as pd
import tensorflow as tf


def expand_prediction_cell(df, colname):
  """Expands all the space separated values in a column into separate columns.
  The assumption is that the column to be expanded is the right-most one.
  Also, since the 5 values being expanded are label, x min, y min, x max, y max,
  those groups will also be put under column names that represent that more explicitly.
  """
  df = df[df.columns[:-1]].join(df[colname].str.split(' ', expand=True))

  renamed_cols = list(df.columns[3:])
  labels = ['lb', 'xmin', 'ymin', 'xmax', 'ymax']

  for i in range(int(len(renamed_cols) / 5)):
    renamed_cols[i * 5:i * 5 + 5] = [lb + str(i) for lb in labels]

  df.columns = list(df.columns[0:3]) + renamed_cols
  num_cols = (df.columns.shape[0] - 3) // 5
  for i in range(num_cols):
    df['xmin{}'.format(i)] = df['xmin{}'.format(i)].astype(float) / df.Width.astype(int)
    df['ymin{}'.format(i)] = df['ymin{}'.format(i)].astype(float) / df.Height.astype(int)
    df['xmax{}'.format(i)] = df['xmax{}'.format(i)].astype(float) / df.Width.astype(int)
    df['ymax{}'.format(i)] = df['ymax{}'.format(i)].astype(float) / df.Height.astype(int)

  return df


def parse_synset_mapping(path):
  """Parse the synset mapping file into a dictionary mapping <synset_id>:[<synonyms in English>]
  This assumes an input file formatted as:
      <synset_id> <category>, <synonym...>
  Example:
      n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
  """
  synset_map = {}
  with open(path, 'r') as fp:
    lines = fp.readlines()
    for line in lines:
      parts = line.split(' ')
      synset_map[parts[0]] = [label.strip() for label in ' '.join(parts[1:]).split(',')]
    return synset_map


def generate_synset_to_int_mapping(synset_mapping):
  synset_to_int_map = {}
  for index, (key, val) in enumerate(synset_mapping.items()):
    synset_to_int_map[key] = index
  return synset_to_int_map


def generate_int_to_synset_mapping(synset_mapping):
  int_to_synset_map = {}
  for index, (key, val) in enumerate(synset_mapping.items()):
    int_to_synset_map[index] = key
  return int_to_synset_map


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
