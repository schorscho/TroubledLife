import tensorflow as tf
import numpy as np
import datetime

# We want variable policy lengths in the output - how?
# Noise as input - but shall we also indicate a policy length, or can the network decide as to the policy runtime and
# the generation of premium adaptation biz proc?

def generator(z, n_outputs):
    n_layers = 3
    n_neurons = 200

    with tf.variable_scope("generator"):
        layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                              activation=tf.nn.relu)
                  for layer in range(n_layers)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=False)
        outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, z, dtype=tf.float32)

        logits = tf.layers.dense(outputs, n_outputs)

    return logits


def generator_loss(logits, batch_size, n_outputs):
    helper = np.ones((batch_size, n_outputs))
    helper[:, 0] = 1

    ideal_logits = tf.constant(helper, name='ideal_logits', dtype=tf.float32)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=ideal_logits))

    #low = tf.zeros_like(logits) + 100000.0
    #high = low + 50000.0

    #g_loss = tf.cast(tf.less(policies, low), tf.float32)# + tf.greater(policies, high)
    #loss = tf.reduce_mean(tf.abs(logits[:, 0, 0] - 100000.0))

    return loss


def generator_trainer(learning_rate, loss):
    trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss)#, var_list=g_vars)

    return trainer


def discriminator(X, n_outputs, reuse=False):
    n_layers = 3
    n_neurons = 200

    with tf.variable_scope("discriminator"):
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                              activation=tf.nn.relu,
                                              reuse=reuse)
                  for layer in range(n_layers)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=False)
        outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

        # We feed all layers' states (after the last timestep) into a fully connected layer of n_outputs neurons
        # (1 per class). Softmax layer is next
        logits = tf.layers.dense(states, n_outputs)
        y_pred = tf.argmax(tf.nn.softmax(logits), axis=1)

    return logits, y_pred


def discriminator_loss(logits, y):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    # Loss function and Adam Optimizer
    loss = tf.reduce_mean(tf.cast(xentropy, tf.float32))

    return loss


def discriminator_trainer(learning_rate, loss):
    trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return trainer


g_learning_rate = 0.001
d_learning_rate = 0.0001
max_policy_history_length = 15
z_dimensions = 100
batch_size = 200
g_n_outputs = 2
d_n_inputs = 2

tf.reset_default_graph()
tf.set_random_seed(42)

seq_length = tf.placeholder(tf.int32, [None], name="seq_length")
z_placeholder = tf.placeholder(tf.float32, [None, max_policy_history_length, z_dimensions], name='z_placeholder')
X_placeholder = tf.placeholder(tf.float32, [None, max_policy_history_length, d_n_inputs], name="X_placeholder")

g_data = generator(z_placeholder, n_outputs=g_n_outputs)

y_pred_X = discriminator(X_placeholder, max_policy_history_length)
logits_g, _ = discriminator(g_data, max_policy_history_length, reuse=True)

g_loss = generator_loss(logits_g, batch_size, max_policy_history_length)
g_trainer = generator_trainer(g_learning_rate, g_loss)

#d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dx, tf.ones_like(Dx)))
#d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dg, tf.zeros_like(Dg)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


for i in range(100000):
    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, max_policy_history_length, z_dimensions])

    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

    if i % 100 == 0:
        # Every 100 iterations, show a generated image
        print("Iteration:", i, "at", datetime.datetime.now())
        z_batch = np.random.normal(0, 1, size=[1, max_policy_history_length, z_dimensions])
        policy = sess.run(g_data, {z_placeholder: z_batch})

        print("Policy:", policy[:, 0, 0], policy[:, 0, 0].shape)