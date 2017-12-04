import tensorflow as tf
from tensorflow.python.layers.core import Dense


def encoder(n_layers, n_neurons, n_latent, X, seq_length):
    with tf.variable_scope("P_Encoder") as scope:
        encoder_cells = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) for layer in
                         range(n_layers)]
        encoder_multi_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells, state_is_tuple=False)
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=encoder_multi_cell, inputs=X, sequence_length=seq_length,
                                                            dtype=tf.float32)

    print("encoder states shape:", encoder_states.shape)

    with tf.variable_scope("P_Encoder_2_Latent") as scope:
        W = tf.get_variable('W', [n_neurons * n_layers, n_latent])
        b = tf.get_variable('b',[n_latent])
        latent_vector = tf.nn.xw_plus_b(encoder_states, W, b)

    print("latent_vector shape:", latent_vector.shape)

    return latent_vector


def decoder(n_latent, n_layers, n_neurons, n_outputs, latent_vector, X, seq_length, training, reuse):
    with tf.variable_scope("Latent_2_P_Decoder") as scope:
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        W = tf.get_variable('W', [n_latent, n_neurons])
        b = tf.get_variable('b', [n_neurons])
        state_input = tf.nn.xw_plus_b(latent_vector, W, b)

        print("state_input shape:", state_input.shape)

    with tf.variable_scope("P_Decoder") as scope:
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        decoder_cells = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) for layer in
                         range(n_layers)]
        decoder_multi_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells, state_is_tuple=False)

        decoder_initial_state = tf.concat(values=[state_input] * n_layers, axis=1)

        print("decoder initial state shape: ", decoder_initial_state.shape)

        # during training, we never sample from the outputs, during inference, we always sample from the outputs and never read the inputs
        # if (training):
        #     decoder_helper = tf.contrib.seq2seq.TrainingHelper(
        #         inputs=X, sequence_length=seq_length)
        # else:
        decoder_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
            inputs=X, sequence_length=seq_length, sampling_probability=1.0)# 0.0 if training else 1.0)

        # Decoder
        output_layer = Dense(units=n_outputs, activation=None, use_bias=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_multi_cell, helper=decoder_helper, initial_state=decoder_initial_state, output_layer=output_layer)
        # Dynamic decoding
        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

        print("decoder outputs rnn_output shape:", decoder_outputs.rnn_output.shape)

        #decoder_sequence = tf.layers.dense(inputs=decoder_outputs.rnn_output, units=n_outputs, activation=None, name="Output_Sequence")

        #print("decoder sequence shape:", decoder_sequence.shape)

        return decoder_outputs.rnn_output
