#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

import os
import numpy as np
from numpy import genfromtxt
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from random import randint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)
no_of_classes = 6

def bias_init_fn_1(shape,partition_info, dtype=None):
  bias_init_np = np.loadtxt('bias_arr.txt')
  bias_init_np = bias_init_np.astype(np.float32)
  return (bias_init_np)

def weight_init_fn_1(shape,partition_info, dtype=None):
  weight_init_np = np.loadtxt('weight_arr.txt')
  weight_init_np = np.reshape(weight_init_np, [-1, 4096])
  weight_init_np = weight_init_np.astype(np.float32)
  print (weight_init_np.shape)
  return (weight_init_np)

def bias_init_fn_2(shape,partition_info, dtype=None):
  bias_init_np = np.loadtxt('bias_arr_fc7.txt')
  bias_init_np = bias_init_np.astype(np.float32)
  return (bias_init_np)

def weight_init_fn_2(shape,partition_info, dtype=None):
  weight_init_np = np.loadtxt('weight_arr_fc7.txt')
  weight_init_np = np.reshape(weight_init_np, [-1, 4096])
  weight_init_np = weight_init_np.astype(np.float32)
  print (weight_init_np.shape)
  return (weight_init_np)  

def boat_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 25088])

  no_of_neural = 4096
  dropout_rate = 0.5
  #dense layer #1
  dense1 = tf.layers.dense(inputs=input_layer, units=no_of_neural, activation=tf.nn.relu, kernel_initializer = weight_init_fn_1, bias_initializer= bias_init_fn_1)
  #dropout #1
  dropout1 = tf.layers.dropout(inputs=dense1, rate=dropout_rate, training=mode == learn.ModeKeys.TRAIN)
  #dense layer #2
  dense2 = tf.layers.dense(inputs=dropout1, units=no_of_neural, activation=tf.nn.relu, kernel_initializer = weight_init_fn_2, bias_initializer= bias_init_fn_2)
  #dropout #2
  dropout2 = tf.layers.dropout(inputs=dense2, rate=dropout_rate, training=mode == learn.ModeKeys.TRAIN)
  #logit for softmax
  logits = tf.layers.dense(inputs=dropout2, units=no_of_classes)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=no_of_classes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
  # Load training and eval data
  data = np.loadtxt('new_class_bouding_1/feature.txt')
  label_data = np.loadtxt('new_class_bouding_1/label.txt')

  train_data = data[:50]
  train_data = np.array(train_data)
  
  train_labels = label_data[:50]
  train_labels = np.array(train_labels)

  eval_data = data[50:]
  eval_data = np.array(eval_data)

  eval_labels = label_data[50:]
  eval_labels = np.array(eval_labels)

  print (train_data.shape)
  print (train_labels.shape)
  # Create the Estimator
  mnist_classifier = learn.Estimator(
      model_fn=boat_model_fn, model_dir="boat_models")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  mnist_classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=50,
      steps=10000,
      monitors=None)

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # Evaluate the model and print results
  eval_results = mnist_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()