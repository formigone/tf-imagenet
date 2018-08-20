import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

import inception_v2_bn_adam as inception

flags = argparse.ArgumentParser()
flags.add_argument('--model_dir', type=str, default='', help='Path to saved_model')
flags.add_argument('--input_img', type=str, default='', help='Path to image to visualize')
flags.add_argument('--width', type=int, default=224, help='New image width after resize')
flags.add_argument('--height', type=int, default=224, help='New image height after resize')

args = flags.parse_args()

estimator = tf.estimator.Estimator(model_dir=args.model_dir, model_fn=inception.model_fn, params={})

# [7, 7, 3, 64]
weights = estimator.get_variable_value('Conv2d_1a_7x7/weights')

# [64]
biases = estimator.get_variable_value('Conv2d_1a_7x7/biases')

X = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
conv2d = tf.nn.conv2d(X, weights, strides=[1, 2, 2, 1], padding='SAME')
b = tf.placeholder(tf.float32, shape=[64])
conv2d_bias = tf.nn.bias_add(conv2d, b)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  img = cv2.imread(args.input_img)
  img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img.astype(np.float32)
  img = img.reshape(1, args.width, args.height, 3)

  # [1, 224, 224, 64]
  features = sess.run(conv2d, feed_dict={X: img})
  features = sess.run(conv2d_bias, feed_dict={conv2d: features, b: biases})
  features = sess.run(tf.nn.relu(features))

  path = '../feature-maps-bn/{}'.format(int(time.time()))
  os.makedirs(path)

  for index in range(64):
    w = np.array(features[0, :, :, index])
    plt.imsave('{}/feat-{}.png'.format(path, index), w)
