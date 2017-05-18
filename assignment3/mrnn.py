#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import logging
import tensorflow as tf

logger = logging.getLogger("hw3.q3.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class MultiplicativeLSTMCell(tf.contrib.rnn.RNNCell):
    """Wrapper around our mLSTM cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size, forget_bias=1.0):
        self._input_size = input_size
        self._state_size = state_size
        self._forget_bias = forget_bias

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._state_size, self._state_size)

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            c, h = state
            nin = self._input_size
            ndim = self._state_size
            wx = tf.get_variable("wx", [nin, ndim * 4])
            wh = tf.get_variable("wh", [ndim, ndim * 4])
            wmx = tf.get_variable("wmx", [nin, ndim])
            wmh = tf.get_variable("wmh", [ndim, ndim])
            b = tf.get_variable("b", [ndim * 4], initializer=tf.constant_initializer(0.0))

            m = tf.matmul(inputs, wmx) * tf.matmul(h, wmh)
            z = tf.matmul(inputs, wx) + tf.matmul(m, wh) + b
            i, f, o, u = tf.split(value=z, axis=1, num_or_size_splits=4)
            c = c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * tf.tanh(u)
            h = tf.nn.sigmoid(o) * tf.tanh(c)

        return h, tf.contrib.rnn.LSTMStateTuple(c, h)

class MultiplicativeGRUCell(tf.contrib.rnn.RNNCell):
    """Wrapper around our mLSTM cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self._input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            h = state
            nin = self._input_size
            ndim = self._state_size
            wx = tf.get_variable("wx", [nin, ndim * 2])
            wh = tf.get_variable("wh", [ndim, ndim * 2])
            wmx = tf.get_variable("wmx", [nin, ndim])
            wmh = tf.get_variable("wmh", [ndim, ndim])
            wcx = tf.get_variable("wcx", [nin, ndim])
            wch = tf.get_variable("wch", [ndim, ndim])
            b = tf.get_variable("b", [ndim * 2], initializer=tf.constant_initializer(1.0))

            m = tf.matmul(inputs, wmx) * tf.matmul(h, wmh)
            z = tf.nn.sigmoid(tf.matmul(inputs, wx) + tf.matmul(m, wh) + b)
            r, u = tf.split(value=z, axis=1, num_or_size_splits=2)
            c = tf.tanh(tf.matmul(inputs, wcx) + tf.matmul(r * h, wch))
            h = u * h + (1 - u) * c

        return h, h
