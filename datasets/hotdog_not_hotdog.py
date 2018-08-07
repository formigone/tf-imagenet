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

FLAGS = flags.FLAGS

categories = [
  # Food that tastes good
  'pretzel',
  'chocolate sauce',
  'ice cream',
  'French loaf',
  'dough',
  'guacamole',
  'pizza',
  'cheeseburger',
  'carbonara',
  'bagel',
  'trifle',
  'ice lolly',
  'burrito',
  'potpie',
  'hotdog',
  'mashed potato',

  # Fruits
  # 'orange',
  # 'pineapple',
  # 'lemon',
  # 'strawberry',
  # 'banana',
  # 'jackfruit',

  # Boring stuff
  # 'bell pepper',
  # 'corn',
  # 'buckeye',
  # 'custard apple',
  # 'head cabbage',
  # 'spaghetti squash',
  # 'acorn',
  # 'artichoke',
  # 'zucchini',
  # 'rotisserie',
  # 'consomme',
]

print('Parsing synset mapping...')
synset_mapping = utils.parse_synset_mapping_with(FLAGS.synset_mapping, categories)

print(synset_mapping)
exit(0)
