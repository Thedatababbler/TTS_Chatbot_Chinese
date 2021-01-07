"""Attention classes for seq2seq model."""

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import \
    BahdanauAttention, BahdanauMonotonicAttention
from tensorflow.python.ops import nn_ops
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from modules.layers import *
from modules.ops import *
import math
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _bahdanau_score
import functools


class BasicAttentionComputer():
    """
    Basic class for computing attention contexts and alignments.
    This is modified from tensorflow implementation.
    """

    def __init__(self,
                 attention_mechanism,
                 attention_layer=None,
                 memory=None,
                 attention_mask=None):
        super(BasicAttentionComputer, self).__init__()
        if attention_mechanism is None and memory is None:
            raise ValueError(
                "attention_mechanism and memory cannot be all none.")
        self.attention_mechanism = attention_mechanism
        self.attention_layer = attention_layer
        self.attention_mask = attention_mask
        if memory is None:
            self.memory = attention_mechanism.values
        else:
            self.memory = memory
        self.batch_size = tf.shape(self.memory)[0]

    def __call__(self, cell_output, attention_state):
        alignments, next_attention_state = self.attention_mechanism(
            cell_output, state=attention_state)
        if self.attention_mask is not None:
            alignments = alignments * self.attention_mask
            alignments = alignments / \
                         tf.clip_by_value(math_ops.reduce_sum(
                             alignments, axis=1, keep_dims=True), 1.0E-20, 10.0)

        # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
        expanded_alignments = array_ops.expand_dims(alignments, 1)
        context = math_ops.matmul(expanded_alignments, self.memory)
        context = array_ops.squeeze(context, [1])

        if self.attention_layer is not None:
            attention = self.attention_layer(
                array_ops.concat([cell_output, context], 1))
        else:
            attention = context

        return attention, alignments, next_attention_state


class GMMAttentionComputer():
    """GMM attention"""

    def __init__(self,
                 memory,
                 input_length,
                 num_mixture=3,
                 scope='gmm_attention'):
        super(GMMAttentionComputer, self).__init__()
        self.memory = memory
        self.input_length = input_length
        self.num_mixture = num_mixture
        self.batch_size = get_tensor_shape(memory)[0]
        self.memory_length = get_tensor_shape(memory)[1]
        self.scope = scope

    def __call__(self, cell_output, prev_mu):
        with tf.variable_scope(self.scope):
            # cell_output: [B, D_c]
            mlp_out = tf.layers.dense(cell_output, units=256, activation=tf.nn.relu)
            mlp_out = tf.layers.dense(mlp_out, units=3 * self.num_mixture)  # [B, 3 * num_mixture]
            sigma_hat, Delta_hat, omega_hat = tf.split(mlp_out, 3, axis=1)  # [B, num_mixture]

            # v0 version
            omega = tf.exp(omega_hat)  # [B, num_mixture]
            Delta = tf.exp(Delta_hat)  # [B, num_mixture]
            sigma = tf.sqrt(tf.exp(-sigma_hat) / 2)  # [B, num_mixture]
            Z = 1
            mu = prev_mu + Delta  # [B, 1] + [B, num_mixture] = [B, num_mixture]

            omega = tf.expand_dims(omega, axis=2)  # [B, num_mixture, 1]
            sigma = tf.expand_dims(sigma, axis=2)  # [B, num_mixture, 1]
            mu = tf.expand_dims(mu, axis=2)  # [B, num_mixture, 1]

            j = tf.range(self.memory_length)  # [T]
            j = tf.reshape(j, (1, 1, self.memory_length))  # [1, 1, T]
            j = tf.tile(j, (self.batch_size, self.num_mixture, 1))  # [B, num_mixture, T]
            j = tf.cast(j, tf.float32)  # [B, num_mixture, T]

            alignments = (omega / Z) * tf.exp(- (j - mu) ** 2 / (2 * sigma ** 2 + 1e-8))  # [B, num_mixture, T]
            alignments = tf.reduce_sum(alignments, axis=1)  # [B, T]

            mask = tf.cast(tf.sequence_mask(self.input_length,
                                            maxlen=self.memory_length),
                           tf.float32)  # [B, T]
            memory = self.memory * tf.expand_dims(mask, 2)  # [B, T, Dm] * [B, T, 1]
            # self.memory:
            attention = math_ops.matmul(tf.expand_dims(alignments, 1), memory)  # [B, 1, T] * [B, T, Dm] -> [B, 1, Dm]
            attention = tf.squeeze(attention, 1)

        return attention, alignments, tf.squeeze(mu, 2)  # [B, 1, Dm], [B, T], [B, 3]
