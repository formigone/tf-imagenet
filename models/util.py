import tensorflow as tf


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.
  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.
  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]
  Returns:
    a tensor with the kernel size.
  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                        tf.minimum(shape[2], kernel_size[1])])
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [
      min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
    ]
  return kernel_size_out


def inception_block(prev, t1x1=2, t3x3=2, t5x5=2, tmp=2, resize=None, norm=True, training=True, name='incep'):
  '''

  :param prev:
  :param t1x1:
  :param t3x3:
  :param t5x5:
  :param tmp:
  :param resize: A tuple where the first element is the argument for `pool_size` and the second for `strides`.
  :param norm:
  :param training:
  :param name:
  :return:
  '''

  with tf.variable_scope(name):
    if resize is not None:
      with tf.variable_scope('downsample'):
        prev = tf.layers.max_pooling2d(prev, pool_size=resize[0], strides=resize[1], name='max_pool')
    with tf.variable_scope('1x1_conv'):
      tower_1x1 = tf.layers.conv2d(prev, filters=t1x1, kernel_size=1, padding='same', activation=None, name='conv_1x1')
      if norm:
        tower_1x1 = tf.layers.batch_normalization(tower_1x1, training=training, name='batch_norm')
      tower_1x1 = tf.nn.relu(tower_1x1, name='act')

    with tf.variable_scope('3x3_conv'):
      tower_3x3 = tf.layers.conv2d(prev, filters=t3x3, kernel_size=1, padding='same', activation=None, name='conv_1x1')
      if norm:
        tower_3x3 = tf.layers.batch_normalization(tower_3x3, training=training, name='batch_norm')
      tower_3x3 = tf.nn.relu(tower_3x3, name='act')

      tower_3x3 = tf.layers.conv2d(tower_3x3, filters=t3x3, kernel_size=3, padding='same', activation=None,
                                   name='conv_3x3')
      if norm:
        tower_3x3 = tf.layers.batch_normalization(tower_3x3, training=training, name='batch_norm2')
      tower_3x3 = tf.nn.relu(tower_3x3, name='act')
      tf.summary.image('conv_3x3', tower_3x3[:, :, :, 0:3], max_outputs=4)

    with tf.variable_scope('5x5_conv'):
      tower_5x5 = tf.layers.conv2d(prev, filters=t5x5, kernel_size=1, padding='same', activation=None, name='conv_1x1')
      if norm:
        tower_5x5 = tf.layers.batch_normalization(tower_5x5, training=training, name='batch_norm')
      tower_5x5 = tf.nn.relu(tower_5x5, name='act')

      tower_5x5 = tf.layers.conv2d(tower_5x5, filters=t5x5, kernel_size=3, padding='same', activation=None,
                                   name='conv_5x5')
      if norm:
        tower_5x5 = tf.layers.batch_normalization(tower_5x5, training=training, name='batch_norm2')
      tower_5x5 = tf.nn.relu(tower_5x5, name='act')
      tf.summary.image('conv_5x5', tower_5x5[:, :, :, 0:3], max_outputs=4)

      tower_5x5 = tf.layers.conv2d(tower_5x5, filters=t5x5, kernel_size=3, padding='same', activation=None,
                                   name='conv2_5x5')
      if norm:
        tower_5x5 = tf.layers.batch_normalization(tower_5x5, training=training, name='batch_norm3')
      tower_5x5 = tf.nn.relu(tower_5x5, name='act')
      tf.summary.image('conv2_5x5', tower_5x5[:, :, :, 0:3], max_outputs=4)

    with tf.variable_scope('maxpool'):
      tower_mp = tf.layers.max_pooling2d(prev, pool_size=3, strides=1, padding='same', name='3x3_maxpool')
      tower_mp = tf.layers.conv2d(tower_mp, filters=tmp, kernel_size=1, padding='same', activation=None,
                                  name='conv_1x1')
      if norm:
        tower_mp = tf.layers.batch_normalization(tower_mp, training=training, name='batch_norm')
        tower_mp = tf.nn.relu(tower_mp, name='act')

    return tf.concat([tower_1x1, tower_3x3, tower_5x5, tower_mp], axis=3)


def inception_block_v2(prev, t1x1=2, t3x3=(2, 2), t5x5=(2, 2), pool_proj=2, norm=True, training=True, name='incep'):
  '''

  :param prev:
  :param t1x1:
  :param t3x3: Tuple with (reduce, filters)
  :param t5x5: Tuple with (reduce, filters)
  :param pool_proj:
  :param norm:
  :param training:
  :param name:
  :return:
  '''

  with tf.variable_scope(name):
    with tf.variable_scope('1x1_conv'):
      tower_1x1 = tf.layers.conv2d(prev, filters=t1x1, kernel_size=1, padding='same', activation=None, name='conv_1x1')
      if norm:
        tower_1x1 = tf.layers.batch_normalization(tower_1x1, training=training, name='batch_norm')
      tower_1x1 = tf.nn.relu(tower_1x1, name='act')

    with tf.variable_scope('3x3_conv'):
      tower_3x3 = tf.layers.conv2d(prev, filters=t3x3[0], kernel_size=1, padding='same', activation=None,
                                   name='conv_1x1')
      if norm:
        tower_3x3 = tf.layers.batch_normalization(tower_3x3, training=training, name='batch_norm')
      tower_3x3 = tf.nn.relu(tower_3x3, name='act')

      tower_3x3 = tf.layers.conv2d(tower_3x3, filters=t3x3[1], kernel_size=3, padding='same', activation=None,
                                   name='conv_3x3')
      if norm:
        tower_3x3 = tf.layers.batch_normalization(tower_3x3, training=training, name='batch_norm2')
      tower_3x3 = tf.nn.relu(tower_3x3, name='act')
      tf.summary.image('conv_3x3', tower_3x3[:, :, :, 0:3], max_outputs=4)

    with tf.variable_scope('5x5_conv'):
      tower_5x5 = tf.layers.conv2d(prev, filters=t5x5[0], kernel_size=1, padding='same', activation=None,
                                   name='conv_1x1')
      if norm:
        tower_5x5 = tf.layers.batch_normalization(tower_5x5, training=training, name='batch_norm')
      tower_5x5 = tf.nn.relu(tower_5x5, name='act')

      tower_5x5 = tf.layers.conv2d(tower_5x5, filters=t5x5[1], kernel_size=3, padding='same', activation=None,
                                   name='conv_5x5')
      if norm:
        tower_5x5 = tf.layers.batch_normalization(tower_5x5, training=training, name='batch_norm2')
      tower_5x5 = tf.nn.relu(tower_5x5, name='act')
      tf.summary.image('conv_5x5', tower_5x5[:, :, :, 0:3], max_outputs=4)

      tower_5x5 = tf.layers.conv2d(tower_5x5, filters=t5x5[1], kernel_size=3, padding='same', activation=None,
                                   name='conv2_5x5')
      if norm:
        tower_5x5 = tf.layers.batch_normalization(tower_5x5, training=training, name='batch_norm3')
      tower_5x5 = tf.nn.relu(tower_5x5, name='act')
      tf.summary.image('conv2_5x5', tower_5x5[:, :, :, 0:3], max_outputs=4)

    with tf.variable_scope('maxpool'):
      tower_mp = tf.layers.max_pooling2d(prev, pool_size=3, strides=1, padding='same', name='3x3_maxpool')
      tower_mp = tf.layers.conv2d(tower_mp, filters=pool_proj, kernel_size=1, padding='same', activation=None,
                                  name='conv_1x1')
      if norm:
        tower_mp = tf.layers.batch_normalization(tower_mp, training=training, name='batch_norm')
        tower_mp = tf.nn.relu(tower_mp, name='act')

    return tf.concat([tower_1x1, tower_3x3, tower_5x5, tower_mp], axis=3)
