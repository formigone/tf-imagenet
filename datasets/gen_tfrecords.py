import pandas as pd
import cv2
import numpy as np
import tensorflow as tf

import utils

# TODO: Use tf argparser or named args instead
flags = tf.app.flags
flags.DEFINE_string('input_csv', '', 'Input file to parse, with columns')
flags.DEFINE_string('output_tfrecord', '', 'Output tfrecord')
flags.DEFINE_string('synset_mapping', '', 'Path to synset mapping file')
flags.DEFINE_integer('width', 299, 'New image width after resize')
flags.DEFINE_integer('height', 299, 'New image height after resize')
flags.DEFINE_boolean('training', True, 'Whether to look for files in the training set directory or not')

FLAGS = flags.FLAGS


files = pd.read_csv(FLAGS.input_csv)
files = utils.expand_prediction_cell(files, 'PredictionString')
files = files.sample(frac=1)

print('Finished parsing input CSV file. Generating TFRecords...')

writer = tf.python_io.TFRecordWriter(FLAGS.output_tfrecord)
i = 0
synset_mapping = utils.parse_synset_mapping(FLAGS.synset_mapping)
synset_to_int = utils.generate_synset_to_int_mapping(synset_mapping)

for index, row in files.iterrows():
  # ImageId, Width, Height, lb0, [xmin0, ymin0, xmax0, ymax0]...

  if i % 1000 == 0:
    print('{}/{}'.format(i, files.shape[0]))

  if FLAGS.training:
    imgId = row.ImageId
    labelId = imgId.split('_')[0]
    path = 'data/train/{}/{}.JPEG'.format(labelId, imgId)
  else:
    labelId = row.lb0
    path = 'data/val/{}.JPEG'.format(row.ImageId)

  img = cv2.imread(path)
  img = cv2.resize(img, (FLAGS.width, FLAGS.height), interpolation=cv2.INTER_CUBIC)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img.astype(np.float32)

  feature = {
    'img': utils._bytes_feature(tf.compat.as_bytes(img.tostring())),
    'class': utils._int64_feature(synset_to_int[labelId]),
    # 'width': utils._int64_feature(row.Width),
    # 'height': utils._int64_feature(row.Height),
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature))
  writer.write(example.SerializeToString())

  i += 1

writer.close()
print('{}/{}'.format(i, files.shape[0]))
