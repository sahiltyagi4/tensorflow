#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
import os
import json
import sys

from official.mnist import dataset
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
#from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers

LEARNING_RATE = 1e-4
#distribution_strategy = distribution_utils.getPSdistribution_strategy()

def create_model(data_format):
  # with distribution_strategy.scope():
  if True:
    if data_format == 'channels_first':
      input_shape = [1, 28, 28]
    else:
      assert data_format == 'channels_last'
      input_shape = [28, 28, 1]

    l = tf.keras.layers
    max_pool = l.MaxPooling2D(
        (2, 2), (2, 2), padding='same', data_format=data_format)
    return tf.keras.Sequential(
      [
        l.Reshape(
          target_shape=input_shape,
          input_shape=(28 * 28,)),
        l.Conv2D(
          32,
          5,
          padding='same',
          data_format=data_format,
          activation=tf.nn.relu),
        max_pool,
        l.Conv2D(
          64,
          5,
          padding='same',
          data_format=data_format,
          activation=tf.nn.relu),
        max_pool,
        l.Flatten(),
        l.Dense(1024, activation=tf.nn.relu),
        l.Dropout(0.4),
        l.Dense(10)
    ])

def define_mnist_flags():
  # with distribution_strategy.scope():
  if True:
    flags_core.define_base()
    flags_core.define_performance(num_parallel_calls=False)
    flags_core.define_image()
    flags.adopt_module_key_flags(flags_core)
    flags_core.set_defaults(data_dir='/root',
                            model_dir='/root',
                            batch_size=100,
                            train_epochs=40)

def model_fn(features, labels, mode, params):
  #with distribution_strategy.scope():
    """The model_fn argument for creating an Estimator."""
    for key in params.keys():
      tf.logging.info('####### for key value is #######' + str(key) + ' ' + str(params[key]))

    tf_config = json.loads(os.environ["TF_CONFIG"])
    tasktype = tf_config["task"]["type"]
    tf.logging.info("^^^^^^ task type is ^^^^^^" + str(tasktype))
    ischief = False
    if tasktype == "chief":
      ischief = True

    model = create_model(params['data_format'])
    image = features
    if isinstance(image, dict):
      image = features['image']

    if mode == tf.estimator.ModeKeys.PREDICT:
      logits = model(image, training=False)
      predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
      }
    
      return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })
  
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
      logits = model(image, training=True)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
      accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))

      # Name tensors to be logged with LoggingTensorHook.
      tf.identity(LEARNING_RATE, name='learning_rate')
      tf.identity(loss, name='cross_entropy')
      tf.identity(accuracy[1], name='train_accuracy')

      #tensors_to_log = {'learning_rate': LEARNING_RATE, 'cross_entropy': loss, 'train_accuracy': accuracy[1]}
      tensors_to_log = {'cross_entropy': loss, 'train_accuracy': accuracy[1]}
      logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
      train_hooks = [logging_hook]

      if ischief:
        optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate = 3)
        sync_replicas_hook = optimizer.make_session_run_hook(ischief)
        train_hooks.append(sync_replicas_hook)

      # Save accuracy scalar to Tensorboard output.
      tf.summary.scalar('train_accuracy', accuracy[1])
      tf.summary.scalar('loss', loss)

      # model_params = tf.trainable_variables()
      # compute_grads = optimizer.compute_gradients(loss, model_params)
      # for g,v in compute_grads:
      # 	tf.summary.histogram(v.name + '_ORG', g)




      return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        training_hooks=train_hooks,
        train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    if mode == tf.estimator.ModeKeys.EVAL:
      logits = model(image, training=False)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
      loss = tf.reduce_sum(loss) * (1. /flags_obj.batch_size)
      accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

      tf.summary.scalar('eval_accuracy', accuracy[1])
      tf.summary.scalar('eval_loss', loss)

      return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        # evaluation_hooks=,
        eval_metric_ops={
          'accuracy':accuracy,
        })


def run_mnist(flags_obj):
  # with distribution_strategy.scope():
  if True:
    """Run MNIST training and eval loop.
    Args:flags_obj: An object containing parsed flag values."""
    model_helpers.apply_clean(flags_obj)
    model_function = model_fn

    session_config = tf.ConfigProto(
      inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
      intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
      allow_soft_placement=True)

    tf.logging.info('$$$$$$$$$$ batch_size: $$$$$$$$$$$$$$$' + str(flags_obj.batch_size))
    tf.logging.info('$$$$$$$$$ train_epochs: $$$$$$$$$$$$$$$$$$' + str(flags_obj.train_epochs))
    tf.logging.info('$$$$$$ epochs_between_evals: $$$$$$$$$$$' + str(flags_obj.epochs_between_evals))
    tf.logging.info('$$$$$ stop_threshold: $$$$$$$$$$$$$$' + str(flags_obj.stop_threshold))
    tf.logging.info('$$$$$$$$ export_dir: $$$$$$$$$$$' + str(flags_obj.export_dir))
    tf.logging.info('$$$$$$ data directory: $$$$$$$$$' + str(flags_obj.data_dir))

    # ps_strategy = distribution_utils.getPSdistribution_strategy()
    # run_config = tf.estimator.RunConfig(train_distribute = ps_strategy, session_config=session_config, model_dir='/root')
    run_config = tf.estimator.RunConfig(session_config=session_config, model_dir='/root')
    data_format = flags_obj.data_format
    if data_format is None:
      data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')

    mnist_classifier = tf.estimator.Estimator(
      model_fn=model_function,
      config=run_config,
      params={
          'data_format': data_format,
    })



  # Set up training and evaluation input functions.
  def train_input_fn():
    """Prepare data for training."""

    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes use less memory. MNIST is a small
    # enough dataset that we can easily shuffle the full epoch.
    ds = dataset.train(flags_obj.data_dir)
    ds = ds.cache().shuffle(buffer_size=50000).batch(flags_obj.batch_size)

    # Iterate through the dataset a set number (`epochs_between_evals`) of times
    # during each training session.
    
    #ds = ds.repeat(flags_obj.epochs_between_evals)
    ds = ds.repeat()
    return ds

  def eval_input_fn():
    return dataset.test(flags_obj.data_dir).batch(flags_obj.batch_size).repeat().make_one_shot_iterator().get_next()

  # Set up hook that outputs training logs every 100 steps.
  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks, model_dir=flags_obj.model_dir,
      batch_size=flags_obj.batch_size)
  acc_threshold = tf.estimator.experimental.stop_if_higher_hook(estimator=mnist_classifier, metric_name="train_accuracy", threshold=0.99)
  train_hooks.append(acc_threshold)

  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=50000, hooks=train_hooks)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=100)
  # for v in range(flags_obj.train_epochs // flags_obj.epochs_between_evals):
  #   tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
  #   val = tf.convert_to_tensor(v)

  tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

  # Export the model
  if flags_obj.export_dir is not None:
    image = tf.placeholder(tf.float32, [None, 28, 28])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': image,
    })
    mnist_classifier.export_savedmodel(flags_obj.export_dir, input_fn,
                                       strip_default_attrs=True)


def main(_):
  #with distribution_strategy.scope():
  if True:
    run_mnist(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_mnist_flags()
  absl_app.run(main)
