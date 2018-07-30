from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

import trainer
import util

tf.logging.set_verbosity(tf.logging.DEBUG)

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def model_fn(features, labels, mode, params):
  """
  Based on https://github.com/tensorflow/tpu/blob/master/models/experimental/inception/inception_v2_tpu_model.py
  :param features:
  :param labels:
  :param mode:
  :param params:
  :return:
  """
  tf.summary.image('0_input', features, max_outputs=4)

  # 224 x 224 x 3
  end_point = 'Conv2d_1a_7x7'
  net = layers.conv2d(features, 64, [7, 7], stride=2, weights_initializer=trunc_normal(1.0), scope=end_point)
  tf.summary.image('1_{}'.format(end_point), net[:, :, :, 0:3], max_outputs=4)

  # 112 x 112 x 64
  end_point = 'MaxPool_2a_3x3'
  net = layers_lib.max_pool2d(net, [3, 3], scope=end_point, stride=2, padding='SAME')
  tf.summary.image('2_{}'.format(end_point), net[:, :, :, 0:3], max_outputs=4)

  # 56 x 56 x 64
  end_point = 'Conv2d_2b_1x1'
  net = layers.conv2d(net, 64, [1, 1], scope=end_point, weights_initializer=trunc_normal(0.1))
  tf.summary.image('3_{}'.format(end_point), net[:, :, :, 0:3], max_outputs=4)

  # 56 x 56 x 64
  end_point = 'Conv2d_2c_3x3'
  net = layers.conv2d(net, 192, [3, 3], scope=end_point)
  tf.summary.image('4_{}'.format(end_point), net[:, :, :, 0:3], max_outputs=4)

  # 56 x 56 x 192
  end_point = 'MaxPool_3a_3x3'
  net = layers_lib.max_pool2d(net, [3, 3], scope=end_point, stride=2, padding='SAME')
  tf.summary.image('5_{}'.format(end_point), net[:, :, :, 0:3], max_outputs=4)

  # 28 x 28 x 192
  # Inception module.
  end_point = 'Mixed_3b'
  with variable_scope.variable_scope(end_point):
    with variable_scope.variable_scope('Branch_0'):
      branch_0 = layers.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
    with variable_scope.variable_scope('Branch_1'):
      branch_1 = layers.conv2d(net, 64, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
      branch_1 = layers.conv2d(branch_1, 64, [3, 3], scope='Conv2d_0b_3x3')
    with variable_scope.variable_scope('Branch_2'):
      branch_2 = layers.conv2d(net, 64, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
      branch_2 = layers.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
      branch_2 = layers.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
    with variable_scope.variable_scope('Branch_3'):
      branch_3 = layers_lib.avg_pool2d(net, [3, 3], padding='SAME', stride=1, scope='AvgPool_0a_3x3')
      branch_3 = layers.conv2d(branch_3, 32, [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
    net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)

    # 28 x 28 x 256
    end_point = 'Mixed_3c'
    with variable_scope.variable_scope(end_point):
      with variable_scope.variable_scope('Branch_0'):
        branch_0 = layers.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
      with variable_scope.variable_scope('Branch_1'):
        branch_1 = layers.conv2d(net, 64, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_1 = layers.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
      with variable_scope.variable_scope('Branch_2'):
        branch_2 = layers.conv2d(net, 64, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_2 = layers.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
        branch_2 = layers.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
      with variable_scope.variable_scope('Branch_3'):
        branch_3 = layers_lib.avg_pool2d(net, [3, 3], padding='SAME', stride=1, scope='AvgPool_0a_3x3')
        branch_3 = layers.conv2d(branch_3, 64, [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
      net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)

    # 28 x 28 x 320
    end_point = 'Mixed_4a'
    with variable_scope.variable_scope(end_point):
      with variable_scope.variable_scope('Branch_0'):
        branch_0 = layers.conv2d(net, 128, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_0 = layers.conv2d(branch_0, 160, [3, 3], stride=2, scope='Conv2d_1a_3x3')
      with variable_scope.variable_scope('Branch_1'):
        branch_1 = layers.conv2d(net, 64, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_1 = layers.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
        branch_1 = layers.conv2d(branch_1, 96, [3, 3], stride=2, scope='Conv2d_1a_3x3')
      with variable_scope.variable_scope('Branch_2'):
        branch_2 = layers_lib.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='MaxPool_1a_3x3')
      net = array_ops.concat([branch_0, branch_1, branch_2], 3)

    # 14 x 14 x 576
    end_point = 'Mixed_4b'
    with variable_scope.variable_scope(end_point):
      with variable_scope.variable_scope('Branch_0'):
        branch_0 = layers.conv2d(net, 224, [1, 1], scope='Conv2d_0a_1x1')
      with variable_scope.variable_scope('Branch_1'):
        branch_1 = layers.conv2d(net, 64, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_1 = layers.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
      with variable_scope.variable_scope('Branch_2'):
        branch_2 = layers.conv2d(net, 96, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_2 = layers.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
        branch_2 = layers.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0c_3x3')
      with variable_scope.variable_scope('Branch_3'):
        branch_3 = layers_lib.avg_pool2d(net, [3, 3], padding='SAME', stride=1, scope='AvgPool_0a_3x3')
        branch_3 = layers.conv2d(branch_3, 128, [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
      net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)

    # 14 x 14 x 576
    end_point = 'Mixed_4c'
    with variable_scope.variable_scope(end_point):
      with variable_scope.variable_scope('Branch_0'):
        branch_0 = layers.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
      with variable_scope.variable_scope('Branch_1'):
        branch_1 = layers.conv2d(net, 96, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_1 = layers.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
      with variable_scope.variable_scope('Branch_2'):
        branch_2 = layers.conv2d(net, 96, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_2 = layers.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
        branch_2 = layers.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0c_3x3')
      with variable_scope.variable_scope('Branch_3'):
        branch_3 = layers_lib.avg_pool2d(net, [3, 3], padding='SAME', stride=1, scope='AvgPool_0a_3x3')
        branch_3 = layers.conv2d(branch_3, 128, [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
      net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)

    # 14 x 14 x 576
    end_point = 'Mixed_4d'
    with variable_scope.variable_scope(end_point):
      with variable_scope.variable_scope('Branch_0'):
        branch_0 = layers.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
      with variable_scope.variable_scope('Branch_1'):
        branch_1 = layers.conv2d(net, 128, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_1 = layers.conv2d(branch_1, 160, [3, 3], scope='Conv2d_0b_3x3')
      with variable_scope.variable_scope('Branch_2'):
        branch_2 = layers.conv2d(net, 128, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_2 = layers.conv2d(branch_2, 160, [3, 3], scope='Conv2d_0b_3x3')
        branch_2 = layers.conv2d(branch_2, 160, [3, 3], scope='Conv2d_0c_3x3')
      with variable_scope.variable_scope('Branch_3'):
        branch_3 = layers_lib.avg_pool2d(net, [3, 3], padding='SAME', stride=1, scope='AvgPool_0a_3x3')
        branch_3 = layers.conv2d(branch_3, 96, [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
      net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)

    # 14 x 14 x 576
    end_point = 'Mixed_4e'
    with variable_scope.variable_scope(end_point):
      with variable_scope.variable_scope('Branch_0'):
        branch_0 = layers.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
      with variable_scope.variable_scope('Branch_1'):
        branch_1 = layers.conv2d(net, 128, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_1 = layers.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
      with variable_scope.variable_scope('Branch_2'):
        branch_2 = layers.conv2d(net, 160, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_2 = layers.conv2d(branch_2, 192, [3, 3], scope='Conv2d_0b_3x3')
        branch_2 = layers.conv2d(branch_2, 192, [3, 3], scope='Conv2d_0c_3x3')
      with variable_scope.variable_scope('Branch_3'):
        branch_3 = layers_lib.avg_pool2d(net, [3, 3], padding='SAME', stride=1, scope='AvgPool_0a_3x3')
        branch_3 = layers.conv2d(branch_3, 96, [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
      net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)

    # 14 x 14 x 576
    end_point = 'Mixed_5a'
    with variable_scope.variable_scope(end_point):
      with variable_scope.variable_scope('Branch_0'):
        branch_0 = layers.conv2d(net, 128, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_0 = layers.conv2d(branch_0, 192, [3, 3], stride=2, scope='Conv2d_1a_3x3')
      with variable_scope.variable_scope('Branch_1'):
        branch_1 = layers.conv2d(net, 192, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_1 = layers.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
        branch_1 = layers.conv2d(branch_1, 256, [3, 3], stride=2, scope='Conv2d_1a_3x3')
      with variable_scope.variable_scope('Branch_2'):
        branch_2 = layers_lib.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='MaxPool_1a_3x3')
      net = array_ops.concat([branch_0, branch_1, branch_2], 3)

    # 7 x 7 x 1024
    end_point = 'Mixed_5b'
    with variable_scope.variable_scope(end_point):
      with variable_scope.variable_scope('Branch_0'):
        branch_0 = layers.conv2d(net, 352, [1, 1], scope='Conv2d_0a_1x1')
      with variable_scope.variable_scope('Branch_1'):
        branch_1 = layers.conv2d(net, 192, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_1 = layers.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
      with variable_scope.variable_scope('Branch_2'):
        branch_2 = layers.conv2d(net, 160, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_2 = layers.conv2d(branch_2, 224, [3, 3], scope='Conv2d_0b_3x3')
        branch_2 = layers.conv2d(branch_2, 224, [3, 3], scope='Conv2d_0c_3x3')
      with variable_scope.variable_scope('Branch_3'):
        branch_3 = layers_lib.avg_pool2d(net, [3, 3], padding='SAME', stride=1, scope='AvgPool_0a_3x3')
        branch_3 = layers.conv2d(branch_3, 128, [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
      net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)

    # 7 x 7 x 1024
    end_point = 'Mixed_5c'
    with variable_scope.variable_scope(end_point):
      with variable_scope.variable_scope('Branch_0'):
        branch_0 = layers.conv2d(net, 352, [1, 1], scope='Conv2d_0a_1x1')
      with variable_scope.variable_scope('Branch_1'):
        branch_1 = layers.conv2d(net, 192, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_1 = layers.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
      with variable_scope.variable_scope('Branch_2'):
        branch_2 = layers.conv2d(net, 192, [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
        branch_2 = layers.conv2d(branch_2, 224, [3, 3], scope='Conv2d_0b_3x3')
        branch_2 = layers.conv2d(branch_2, 224, [3, 3], scope='Conv2d_0c_3x3')
      with variable_scope.variable_scope('Branch_3'):
        branch_3 = layers_lib.max_pool2d(net, [3, 3], padding='SAME', stride=1, scope='MaxPool_0a_3x3')
        branch_3 = layers.conv2d(branch_3, 128, [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
      net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)

  with variable_scope.variable_scope('Logits'):
    kernel_size = util._reduced_kernel_size_for_small_input(net, [7, 7])
    net = layers_lib.avg_pool2d(net, kernel_size, stride=1, padding='VALID', scope='AvgPool_1a_{}x{}'.format(*kernel_size))

    # 1 x 1 x 1024
    net = layers_lib.dropout(net, keep_prob=params['dropout_keep_prob'], scope='Dropout_1b')
    logits = layers.conv2d(net, params['num_classes'], [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
    if params['spatial_squeeze']:
      logits = array_ops.squeeze(logits, [1, 2], name='SpatialSqueeze')

  predictions = {
    'argmax': tf.argmax(logits, axis=1, name='prediction_classes'),
    'predictions': layers_lib.softmax(logits, scope='Predictions'),
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
  tf.summary.scalar('loss', loss)

  eval_metric_ops = {
    'accuracy_val': tf.metrics.accuracy(labels=labels, predictions=predictions['argmax'])
  }

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
  extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(extra_update_ops):
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

  tf.summary.scalar('accuracy_train', eval_metric_ops['accuracy_val'][1])
  tf.summary.histogram('labels', labels)
  tf.summary.histogram('predictions', predictions['argmax'])

  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
  trainer.run(model_fn=model_fn)
