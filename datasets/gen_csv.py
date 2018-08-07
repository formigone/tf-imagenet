import pandas as pd
import tensorflow as tf

import utils

flags = tf.app.flags
flags.DEFINE_string('input_csv', '', 'Input file to parse, with columns')
flags.DEFINE_string('output_csv', '', 'Output csv')
flags.DEFINE_string('synset_mapping', '', 'Path to synset mapping file')
flags.DEFINE_integer('width', 299, 'New image width after resize')
flags.DEFINE_integer('height', 299, 'New image height after resize')
flags.DEFINE_boolean('training', True, 'Whether to look for files in the training set directory or not')
flags.DEFINE_string('use_only_from_classes', '', 'Optional. When a comma-separated list of synset values is passes, a synset map will be created of only those values.')

FLAGS = flags.FLAGS

df_data = []

print('Parsing synset mapping...')
classes = filter(None, [className.strip() for className in FLAGS.use_only_from_classes.split(',')])

if len(classes) > 0:
  print('Restricting synset to only: {}'.format(classes))

synset_mapping = utils.parse_synset_mapping_with(FLAGS.synset_mapping, classes)

files = pd.read_csv(FLAGS.input_csv)
files = utils.expand_prediction_cell(files, 'PredictionString')
files = files.sample(frac=1)

i = 0

print('Finished parsing input CSV file. Generating CSVs...')

report_every = 1000 if len(classes) == 0 else 50

for index, row in files.iterrows():
  # ImageId, Width, Height, lb0, [xmin0, ymin0, xmax0, ymax0]...

  if i % report_every == 0:
    print('{}/{}'.format(i, files.shape[0]))

  labelId = row.lb0

  if len(classes) > 0 and labelId not in synset_mapping:
    continue

  if FLAGS.training:
    imgId = row.ImageId
    path = 'data/train/{}/{}.JPEG'.format(labelId, imgId)
  else:
    path = 'data/val/{}.JPEG'.format(row.ImageId)

  feature = {
    'img': path,
    'class': synset_mapping[labelId][0],
    # 'width': utils._int64_feature(row.Width),
    # 'height': utils._int64_feature(row.Height),
  }

  df_data.append(feature)

  i += 1

df = pd.DataFrame(df_data, columns=['img', 'class'])
df.to_csv(FLAGS.output_csv, index=False)
print('{}/{}'.format(i, files.shape[0]))
