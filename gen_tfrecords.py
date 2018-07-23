import pandas as pd
import argparse
from PIL import Image
import numpy as np

# TODO: Use tf argparser or named args instead
parser = argparse.ArgumentParser()
parser.add_argument('input_csv', help='Input file to parse, with columns')
parser.add_argument('output_tfrecord', help='Output tfrecord')
parser.add_argument('training', type=bool, help='Whether to look for files in the training set directory or not')

args = parser.parse_args()


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


files = pd.read_csv(args.input_csv)
files = expand_prediction_cell(files, 'PredictionString')
files = files.sample(frac=1)

print('Finished parsing input CSV file. Generating TFRecords...')

for index, row in files.iterrows():
  # ImageId, Width, Height, lb0, [xmin0, ymin0, xmax0, ymax0]...

  # TODO: change this based if it's args.training
  imgId = row.ImageId
  path = 'data/train/{}/{}.JPEG'.format(imgId.split('_')[0], imgId)
  img = Image.open(path)
  img = np.array(img.resize((299, 299))) / 255
  img = img.astype(np.float32)
  print(img.shape)
  print(row.ImageId)

# print(files)
