import tensorflow as tf

import trainer

tf.logging.set_verbosity(tf.logging.DEBUG)


def model_fn(features, labels, mode, params):
  training = mode == tf.estimator.ModeKeys.TRAIN
  x = tf.layers.batch_normalization(features, training=training, name='x_norm__cnn_01')

  x = tf.reshape(x, [-1, params['height'], params['width'], 3])
  tf.summary.image('input', x)

  x = tf.layers.conv2d(x, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv1')
  x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=2, name='pool1')
  tf.summary.image('pool1', x[:, :, :, 0:3])

  flat = tf.contrib.layers.flatten(x, scope='flatten')
  dense = tf.layers.dense(flat, units=64, activation=tf.nn.relu, name='dense')
  dropout = tf.layers.dropout(dense, rate=params['dropout_rate'], training=training, name='dropout')

  logits = tf.layers.dense(dropout, units=params['num_classes'], name='logits')

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
