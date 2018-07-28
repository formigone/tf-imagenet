from __future__ import division

import tensorflow as tf
import pandas as pd
import itertools

from data import gen_input


def parse_args():
  flags = tf.app.flags
  flags.DEFINE_string('train_set', '', 'TFRecord used for training')
  flags.DEFINE_string('val_set', '', 'TFRecord used for evaluation')
  flags.DEFINE_string('model_dir', '', 'Path to saved_model')

  flags.DEFINE_string('mode', 'train', 'Either train or predict')
  flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
  flags.DEFINE_float('dropout', 0.7, 'Dropout percentage')
  flags.DEFINE_bool('spatial_squeeze', True, 'If True, logits is of shape [B, C], if false logits is of shape [B, 1, 1, C], where B is batch_size and C is number of classes.')
  flags.DEFINE_integer('num_classes', 1000, 'Number of classes to classify')
  flags.DEFINE_integer('batch_size', 16, 'Input function batch size')
  flags.DEFINE_integer('buffer_size', 64, 'Input function buffer size')
  flags.DEFINE_integer('img_width', 299, 'Width of input image')
  flags.DEFINE_integer('img_height', 299, 'Height of input image')

  flags.DEFINE_string('predict_set', '', 'TFRecord used for prediction')
  flags.DEFINE_string('predict_csv', '', 'Name of test file')
  flags.DEFINE_integer('max_steps', 10000, 'How many steps to take during training - including current step')
  flags.DEFINE_integer('epochs', 1, 'How many epochs before the input_fn throws OutOfRange exception')

  args = flags.FLAGS
  args._parse_flags()
  return args


def run(model_fn):
  args = parse_args()
  tf.logging.debug('Tensorflow version: {}'.format(tf.__version__))

  model_params = {
    'train_set': args.train_set,
    'val_set': args.val_set,
    'model_dir': args.model_dir,
    'learning_rate': args.learning_rate,
    'dropout_rate': args.dropout,
    'spatial_squeeze': args.spatial_squeeze,
    'num_classes': args.num_classes,
    'width': args.img_width,
    'height': args.img_height,
  }

  tf.logging.debug('Args: {}'.format(args.__flags))
  tf.logging.debug('Model Params: {}'.format(model_params))

  estimator = tf.estimator.Estimator(model_dir=args.model_dir, model_fn=model_fn, params=model_params)

  if args.mode == 'train':
    train_input_fn = gen_input(args.train_set.split(','),
                               batch_size=args.batch_size,
                               buffer_size=args.buffer_size,
                               img_shape=[args.img_height, args.img_width, 3],
                               repeat=args.epochs)
    eval_input_fn = gen_input(args.val_set.split(','),
                              img_shape=[args.img_height, args.img_width, 3])

    tf.logging.info('Training for {}'.format(None if args.max_steps < 1 else args.max_steps))
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=None if args.max_steps < 1 else args.max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  elif args.mode == 'predict':
    input_fn = gen_input(args.predict_set, batch_size=args.batch_size, img_shape=[args.img_height, args.img_width, 3])

    tf.logging.debug('Generating predictions using {}'.format(args.predict_set))
    predictions = estimator.predict(input_fn=input_fn)
    tf.logging.debug('Got predictions')

    # for pred in predictions:
    #   print('{}'.format(pred['cholesterol']))
    # return

    tf.logging.debug('Reading input csv {}'.format(args.predict_csv))
    df = pd.read_csv(args.predict_csv)
    mse = 0
    correct = wrong = max = 0
    min = 999999
    if True:
      # out_fh.write('id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n')
      for pred, (i, row) in itertools.izip(predictions, df.iterrows()):
        val_exact = float(row[-1:])
        margin = val_exact * 0.15  # 10%
        print(pred['cholesterol']);
        if args.label_scale == 0:
          pred_norm = int(pred['cholesterol'] * 210 + 90)
        else:
          pred_norm = int(pred['cholesterol'] * args.label_scale)
        match = False
        if int(val_exact - margin) <= pred_norm <= int(val_exact + margin):
          correct += 1
          match = True
        else:
          wrong += 1

        mse += (pred_norm - val_exact) ** 2
        if pred_norm < min:
          min = pred_norm
        if pred_norm > max:
          max = pred_norm

        print('{} [{}, {}] {} {}'.format(
          int(val_exact),
          int(val_exact - margin),
          int(val_exact + margin),
          pred_norm,
          '' if match else '*'
        ))
      tf.logging.debug('Accuracy {}/{} ({}%)'.format(correct, correct + wrong, correct / (correct + wrong) * 100))
      tf.logging.debug('Min/max {}/{}'.format(min, max))
      tf.logging.debug('MSE {}'.format(mse / (correct + wrong)))
  else:
    raise ValueError('Invalid mode')
