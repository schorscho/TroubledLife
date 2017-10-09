import tensorflow as tf


def generator(noise, n_outputs, seq_length):
    n_layers = 3
    n_neurons = 200

    with tf.variable_scope("t_generator"):
        layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                              activation=tf.nn.relu)
                  for layer in range(n_layers)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=False)
        outputs, states = tf.nn.dynamic_rnn(cell=multi_layer_cell, inputs=noise, sequence_length=seq_length, dtype=tf.float32)

        logits = tf.layers.dense(outputs, n_outputs)

    return logits


def generator_loss(batch_size, logits):
    # discriminator should ideally fall for the fakes, so the discriminator's logits should flag 0
    zeros = tf.zeros(batch_size, tf.int32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=zeros, logits=logits)
    loss = tf.reduce_mean(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, zeros, 1), tf.float32))

    #low = tf.zeros_like(logits) + 100000.0
    #high = low + 50000.0

    #g_loss = tf.cast(tf.less(policies, low), tf.float32)# + tf.greater(policies, high)
    #loss = tf.reduce_mean(tf.abs(logits[:, 0, 0] - 100000.0))

    return loss, accuracy


def generator_trainer(learning_rate, loss):
    tvars = tf.trainable_variables()

    vars = [var for var in tvars if "t_generator" in var.name]

    trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=vars)

    return trainer


def discriminator(X, n_outputs, seq_length, reuse=False):
    n_layers = 3
    n_neurons = 200

    with tf.variable_scope("t_discriminator"):
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                              activation=tf.nn.relu,
                                              reuse=reuse)
                  for layer in range(n_layers)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=False)
        outputs, states = tf.nn.dynamic_rnn(cell=multi_layer_cell, inputs=X, sequence_length=seq_length, dtype=tf.float32)

        # We feed all layers' states (after the last timestep) into a fully connected layer of n_outputs neurons
        # (1 per class). Softmax layer is next
        logits = tf.layers.dense(states, n_outputs)
        y_pred = tf.argmax(tf.nn.softmax(logits), axis=1)

    return logits, y_pred


def discriminator_loss_real(logits, y):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    loss = tf.reduce_mean(tf.cast(loss, tf.float32))

    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))

    return loss, accuracy


def discriminator_loss_fake(batch_size, logits):
    # discriminator should ideally not fall for the fakes

    ones = tf.ones(batch_size, tf.int64)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ones, logits=logits)

    loss = tf.reduce_mean(tf.cast(loss, tf.float32))

    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, ones, 1), tf.float32))

    return loss, accuracy


def discriminator_trainer_real(learning_rate, loss):
    tvars = tf.trainable_variables()

    vars = [var for var in tvars if "t_discriminator" in var.name]

    trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=vars)

    return trainer


def discriminator_trainer_fake(learning_rate, loss):
    tvars = tf.trainable_variables()

    vars = [var for var in tvars if "t_discriminator" in var.name]

    trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=vars)

    return trainer



        # for i in range(100000):
#     # Train generator
#     noise_batch = np.random.normal(0, 1, size=[batch_size, max_policy_history_length, z_dimensions])
#
#     _ = sess.run(g_trainer, feed_dict={noise: noise_batch})
#
#     if i % 100 == 0:
#         # Every 100 iterations, show a generated image
#         print("Iteration:", i, "at", datetime.datetime.now())
#         noise_batch = np.random.normal(0, 1, size=[1, max_policy_history_length, z_dimensions])
#         policy = sess.run(g_data, {noise: noise_batch})
#
#         print("Policy:", policy[:, 0, 0], policy[:, 0, 0].shape)
#


# Train generator and discriminator together
# for i in range(100000):
#     real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
#     z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
#
#     # Train discriminator on both real and fake images
#     _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
#                                            {x_placeholder: real_image_batch, z_placeholder: z_batch})
#
#     # Train generator
#     z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
#     _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})
#
#     if i % 10 == 0:
#         # Update TensorBoard with summary statistics
#         z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
#         summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
#         writer.add_summary(summary, i)
#
#     if i % 100 == 0:
#         # Every 100 iterations, show a generated image
#         print("Iteration:", i, "at", datetime.datetime.now())
#         z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
#         generated_images = generator(z_placeholder, 1, z_dimensions)
#         images = sess.run(generated_images, {z_placeholder: z_batch})
#         plt.imshow(images[0].reshape([28, 28]), cmap='Greys')
#         plt.show()
#
#         # Show discriminator's estimate
#         im = images[0].reshape([1, 28, 28, 1])
#         result = discriminator(x_placeholder)
#         estimate = sess.run(result, {x_placeholder: im})
#         print("Estimate:", estimate)