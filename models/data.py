from __future__ import division

import tensorflow as tf


def gen_input(filename, batch_size=16, repeat=1, buffer_size=1, img_shape=(128, 128, 3)):
  tf.logging.debug('input_fn: {}'.format({
    'batch_size': batch_size,
    'repeat': repeat,
    'buffer_size': buffer_size,
    'input': filename,
  }))

  def decode(line):
    features = {
      'img': tf.FixedLenFeature([], tf.string),
      'class': tf.FixedLenFeature([], tf.int64),
    }

    parsed = tf.parse_single_example(line, features)
    img = tf.decode_raw(parsed['img'], tf.float32)
    img = tf.reshape(img, img_shape)

    return img / 256, parsed['class']

  def input_fn():
    dataset = (tf.data.TFRecordDataset(filename)).map(decode)
    if buffer_size > 1:
      dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat(repeat)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

  return input_fn


def gen_input_from_data(features, feature_scale=255):
  def input_fn():
    return features / feature_scale, None

  return input_fn
