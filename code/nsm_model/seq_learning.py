import tensorflow as tf
from tensorflow.contrib import rnn


def bert_encoding_layer(bert_module, bert_in, args):
    '''
       ........  BERT Encoding layer .......
    :param bert_module:
    :param bert_in:
    :param args:
    :return:
    '''

    bert_input_dict = dict(input_ids=bert_in['input_ids'],
                            input_mask=bert_in['input_mask'],
                            segment_ids=bert_in['segment_ids'])

    # get embeddings ...
    bert_outputs = bert_module(bert_input_dict, signature="tokens", as_dict=True)

    pooled_output = bert_outputs["pooled_output"]
    sequence_output = bert_outputs["sequence_output"]
    return sequence_output, pooled_output


def rnn_encoding_layer(dropout, encoder_embed_input, args, embed_dim, layer_name="rnn_encoding_v1"):
    '''
        RNN Sequence Encoding Layer ..
    '''

    # representation learning layer
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        if args['encoder_type'] in {'rnn_bi', 'rnn_bi+cnn'}:
            # forward cell.
            fw_lstm_cell = rnn.BasicLSTMCell(embed_dim, state_is_tuple=True)  # last dim size
            fw_encoder_cell = tf.contrib.rnn.DropoutWrapper(fw_lstm_cell, output_keep_prob=1.0 - dropout)

            # backward cell.
            bw_lstm_cell = rnn.BasicLSTMCell(embed_dim, state_is_tuple=True)  # last dim size
            bw_encoder_cell = tf.contrib.rnn.DropoutWrapper(bw_lstm_cell, output_keep_prob=1.0 - dropout)

            # # bidirectional RNN
            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_encoder_cell,
                                                                             cell_bw=bw_encoder_cell,
                                                                             inputs=encoder_embed_input,
                                                                             dtype=tf.float32)
            output_fw, output_bw = encoder_outputs
            encoder_all_outputs = tf.concat([output_fw, output_bw], axis=2)

            # Combine forward and backward output.
            fw_output = tf.transpose(output_fw, [1, 0, 2])  # or time major =True
            fw_last_state = tf.gather(fw_output, tf.shape(fw_output)[0] - 1)

            bw_output = tf.transpose(output_bw, [1, 0, 2])  # or time major =True
            bw_last_state = tf.gather(bw_output, tf.shape(bw_output)[0] - 1)

            encoder_state = tf.concat([fw_last_state, bw_last_state], axis=1)
        else:
            # GRU cell.
            gru = rnn.BasicLSTMCell(embed_dim, state_is_tuple=True)  # last dim size
            # drop_gru = gru
            drop_gru = tf.contrib.rnn.DropoutWrapper(gru, output_keep_prob=1.0 - dropout)
            encoder_cell = tf.contrib.rnn.MultiRNNCell([drop_gru] * 1, state_is_tuple=True)
            init_state = encoder_cell.zero_state(tf.shape(encoder_embed_input)[0], tf.float32)

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_embed_input,
                                                               initial_state=init_state, dtype=tf.float32)
            encoder_all_outputs = encoder_outputs
            # Select last output.
            output = tf.transpose(encoder_outputs, [1, 0, 2])  # or time major =True
            encoder_state = tf.gather(output, int(output.get_shape()[0]) - 1)
    return encoder_all_outputs, encoder_state


def cnn_encoding_layer(dropout, encoder_embed_input, args, embed_dim, sequence_length, layer_name="cnn_encoding_1"):
    '''
            CNN Sequence Encoding Layer ..
            encoder_embed_input: 32 x 15 x 300
    '''
    filter_sizes = [2, 3, 4, 5]
    num_filters = 128

    encoder_embed_input_expanded = tf.expand_dims(encoder_embed_input, -1)  # 32 x 15 x 300 x 1

    # representation learning layer
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_pool_size = (sequence_length - filter_size + 1)

                # Convolution Layer
                filter_shape = [filter_size, embed_dim, 1, num_filters]  # 2 x 300 x 1 x 64
                conv_W = tf.get_variable("conv_W_%s" % filter_size, shape=filter_shape,
                                         initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                conv_b = tf.get_variable("conv_b_%s" % filter_size, shape=[num_filters],
                                         initializer=tf.zeros_initializer(), trainable=True)

                conv = tf.nn.conv2d(input=encoder_embed_input_expanded,
                                    filter=conv_W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name="relu")

                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(value=h,
                                        ksize=[1, filter_pool_size, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID", name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, axis=3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        encoder_state = tf.contrib.layers.fully_connected(h_pool_flat, embed_dim,
                                                          activation_fn=None, reuse=tf.AUTO_REUSE,
                                                          scope=layer_name)  # 32 x 600
    return encoder_state