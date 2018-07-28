import tensorflow as tf

import trainer
import util

tf.logging.set_verbosity(tf.logging.DEBUG)


def model_fn(features, labels, mode, params):
  training = mode == tf.estimator.ModeKeys.TRAIN
  x = tf.layers.batch_normalization(features, training=training, name='x_norm__simple_inception')
  tf.summary.image('input', x)

  x = tf.layers.conv2d(x, filters=16, kernel_size=7, activation=None, name='l1')
  x = tf.layers.batch_normalization(x, training=training, name='batch_norm1')
  x = tf.nn.relu(x, name='act1')
  tf.summary.image('1_l1', x[:, :, :, 0:3], max_outputs=4)

  x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, name='maxpool1')
  tf.summary.image('1_maxpool', x[:, :, :, 0:3], max_outputs=4)

  x = tf.layers.conv2d(x, filters=32, kernel_size=3, activation=None, name='l2')
  x = tf.layers.batch_normalization(x, training=training, name='batch_norm2')
  x = tf.nn.relu(x, name='act2')
  tf.summary.image('2_l2', x[:, :, :, 0:3], max_outputs=4)

  x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, name='maxpool2')
  tf.summary.image('2_maxpool', x[:, :, :, 0:3], max_outputs=4)

  x = util.inception_block(x, t1x1=32, t3x3=42, t5x5=16, tmp=16, training=training, name='3_incep_1')
  x = util.inception_block(x, t1x1=64, t3x3=128, t5x5=52, tmp=43, training=training, name='4_incep_2')
  x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, name='maxpool3')
  tf.summary.image('5_maxpool', x[:, :, :, 0:3], max_outputs=4)

  x = util.inception_block(x, t1x1=96, t3x3=128, t5x5=16, tmp=16, training=training, name='6_incep_3')
  x = util.inception_block(x, t1x1=96, t3x3=128, t5x5=16, tmp=16, training=training, name='7_incep_4')
  x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, name='maxpool4')
  tf.summary.image('8_maxpool', x[:, :, :, 0:3], max_outputs=4)

#  x = util.inception_block(x, t1x1=96, t3x3=128, t5x5=16, tmp=16, training=training, name='8_incep_5')
#  x = util.inception_block(x, t1x1=96, t3x3=128, t5x5=16, tmp=16, training=training, name='9_incep_6')
#  x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, name='maxpool5')
#  tf.summary.image('9_maxpool', x[:, :, :, 0:3], max_outputs=4)

  x = util.inception_block(x, t1x1=128, t3x3=256, t5x5=32, tmp=16, training=training, name='10_incep_7')
  x = util.inception_block(x, t1x1=128, t3x3=256, t5x5=32, tmp=16, training=training, name='11_incep_8')
  x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, name='maxpool6')
  tf.summary.image('10_maxpool', x[:, :, :, 0:3], max_outputs=4)

  flat = tf.contrib.layers.flatten(x, scope='flatten')
  dropout = tf.layers.dropout(flat, rate=params['dropout_rate'], training=training, name='dropout')
  fc = tf.layers.dense(dropout, units=1028, name='fc')

  logits = tf.layers.dense(fc, units=params['num_classes'], name='logits')

  predictions = {
    'argmax': tf.argmax(logits, axis=1, name='prediction_classes'),
    'softmax': tf.nn.softmax(logits, name='prediction_softmax')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  onehot_labels = tf.one_hot(indices=labels, depth=params['num_classes'], name='onehot_labels')
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
  tf.summary.scalar('loss', loss)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
  extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(extra_update_ops):
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
  eval_metric_ops = {
    'accuracy_val': tf.metrics.accuracy(labels=labels, predictions=predictions['argmax'])
  }

  tf.summary.histogram('labels', labels)
  tf.summary.scalar('accuracy_train', eval_metric_ops['accuracy_val'][1])

  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
  trainer.run(model_fn=model_fn)
