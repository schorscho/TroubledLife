import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def leaky_relu(features, alpha=0.2, name=None):
    """Compute the Leaky ReLU activation function.
    "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
    AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
    http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
    Args:
    features: A `Tensor` representing preactivation values.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).
    Returns:
    The activation value.
    """
    with ops.name_scope(name, "LeakyRelu", [features, alpha]):
        features = ops.convert_to_tensor(features, name="features")
        alpha = ops.convert_to_tensor(alpha, name="alpha")
        return math_ops.maximum(alpha * features, features)


def generator(Z, n_outputs, seq_length, leaky):
    n_layers = 3
    n_neurons = 300

    with tf.variable_scope("t_generator"):
        layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                              activation=leaky_relu if leaky else tf.nn.relu)
                  for layer in range(n_layers)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=False)
        outputs, states = tf.nn.dynamic_rnn(cell=multi_layer_cell, inputs=Z, sequence_length=seq_length, dtype=tf.float32)

        logits = tf.layers.dense(outputs, n_outputs)

    return logits


def generator_loss(size_batch, Dg):
    # discriminator should ideally fall for the fakes, so the discriminator's logits should flag 0
    zeros = tf.zeros(size_batch, tf.int32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=zeros, logits=Dg)
    loss = tf.reduce_mean(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(Dg, zeros, 1), tf.float32))

    return loss, accuracy


def generator_trainer(learning_rate, loss):
    tvars = tf.trainable_variables()

    vars = [var for var in tvars if "t_generator" in var.name]

    trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=vars)

    return trainer


def discriminator(X, seq_length, n_outputs, leaky, reuse=False):
    n_layers = 3
    n_neurons = 200

    with tf.variable_scope("t_discriminator"):
        if (reuse):
            tf.get_variable_scope().reuse_variables()


        layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                              activation=leaky_relu if leaky else tf.nn.relu)
                  for layer in range(n_layers)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=False)
        outputs, states = tf.nn.dynamic_rnn(cell=multi_layer_cell, inputs=X, sequence_length=seq_length, dtype=tf.float32)

        # We feed all layers' states (after the last timestep) into a fully connected layer of n_outputs neurons
        # (1 per class). Softmax layer is next
        logits = tf.layers.dense(states, n_outputs)
        y_pred = tf.argmax(tf.nn.softmax(logits), axis=1)

    return logits, y_pred


def discriminator_loss_real(Dx, y):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=Dx)

    loss = tf.reduce_mean(tf.cast(loss, tf.float32))

    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(Dx, y, 1), tf.float32))

    return loss, accuracy


def discriminator_loss_fake(size_batch, Dg):
    # discriminator should ideally not fall for the fakes

    ones = tf.ones(size_batch, tf.int64)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ones, logits=Dg)

    loss = tf.reduce_mean(tf.cast(loss, tf.float32))

    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(Dg, ones, 1), tf.float32))

    return loss, accuracy


def discriminator_trainer_real(learning_rate, loss):
    tvars = tf.trainable_variables()

    vars = [var for var in tvars if "t_discriminator" in var.name]

    trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=vars)

    return trainer


