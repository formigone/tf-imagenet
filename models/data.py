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

    # [<tf.Tensor 'Reshape:0' shape=(224, 224, 3) dtype=float32>, <tf.Tensor 'ParseSingleExample/Squeeze_class:0' shape=() dtype=int64>]
    # print([img, parsed['class']])

    return img / 255, parsed['class']

  def decode_csv(line):
    parsed = tf.decode_csv(line, record_defaults=[['r'],[0]])
    path = parsed[0]
    value = tf.read_file(path)
    feature = tf.image.decode_jpeg(value, channels=3)
    feature = tf.image.resize_images(feature, [img_shape[1], img_shape[0]]) / 255

    feature = tf.image.random_brightness(feature, max_delta=32. / 255.)
    feature = tf.image.random_saturation(feature, lower=0.5, upper=1.5)
    feature = tf.image.random_hue(feature, max_delta=0.2)
    feature = tf.image.random_contrast(feature, lower=0.5, upper=1.5)
    feature = tf.image.random_flip_left_right(feature)
    feature = tf.image.random_flip_up_down(feature)

    label = tf.cast(parsed[-1], dtype=tf.int64)

    # [<tf.Tensor 'truediv:0' shape=(224, 224, 3) dtype=float32>, <tf.Tensor 'Cast:0' shape=() dtype=int64>]
    # print([feature, label])

    return feature, label

  def input_fn():
    if filename[0].endswith('.csv'):
      dataset = tf.data.TextLineDataset(filename).skip(1).map(decode_csv)
      # <MapDataset shapes: ((224, 224, 3), ()), types: (tf.float32, tf.int64)>
    else:
      dataset = (tf.data.TFRecordDataset(filename)).map(decode)
      # <MapDataset shapes: ((224, 224, 3), ()), types: (tf.float32, tf.int64)>

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
