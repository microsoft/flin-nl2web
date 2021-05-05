import tensorflow as tf
from tensorflow.contrib import rnn


def word_embedding_layer(enc_vocab_size, embedding_dim, use_pre_train, is_trainable):

    with tf.variable_scope("word_embedding", reuse=tf.AUTO_REUSE):
        embedding_encoder = tf.get_variable("encoder_embedding_word", [enc_vocab_size, embedding_dim],
                                            trainable=is_trainable)

        embedding_placeholder = None
        embedding_init = None

        if use_pre_train:
            embedding_placeholder = tf.placeholder(tf.float32, [enc_vocab_size, embedding_dim])
            embedding_init = embedding_encoder.assign(embedding_placeholder)
    return embedding_encoder, embedding_placeholder, embedding_init


def match_embedding_layer(embedding_dim, is_trainable):
    with tf.variable_scope("Match_embedding", reuse=tf.AUTO_REUSE):
            embedding_encoder = tf.get_variable("encoder_embedding_match", [4, embedding_dim],
                                                trainable=is_trainable)
    return embedding_encoder


def char_embedding_layer(enc_vocab_size, embedding_dim, is_trainable):

    with tf.variable_scope("char_embedding", reuse=tf.AUTO_REUSE):
        embedding_encoder = tf.get_variable("encoder_embedding_char", [enc_vocab_size, embedding_dim],
                                            trainable=is_trainable)
    return embedding_encoder


def ent_embedding_layer(enc_vocab_size, embedding_dim, is_trainable):

    with tf.variable_scope("ent_embedding", reuse=tf.AUTO_REUSE):
        embedding_encoder = tf.get_variable("encoder_embedding_ent", [enc_vocab_size, embedding_dim],
                                            trainable=is_trainable)
    return embedding_encoder


def get_char_embeddings_phrase(dropout, phrase_in_char, char_embed_encoder,
                               phrase_seq_len, word_seq_len, embedding_dim, args, layer_name):
    char_emb_inputs = tf.nn.embedding_lookup(char_embed_encoder, phrase_in_char)  # 32 x 15 x 12 x 100
    char_emb_inputs_2 = tf.reshape(char_emb_inputs, (tf.shape(char_emb_inputs)[0] * phrase_seq_len, word_seq_len,
                                                         embedding_dim))  # 32*15 x 12 x 100

    phrase_char_embs = char_rnn_encoding_layer(dropout, char_emb_inputs_2, embedding_dim, layer_name)  # 32*15 x 100

    char_embs_out = tf.reshape(phrase_char_embs, (tf.shape(char_emb_inputs)[0],
                                                           phrase_seq_len, embedding_dim))  # 32 x 15 x 100
    return char_embs_out


def char_rnn_encoding_layer(dropout, char_encoder_embed_input, embed_dim, layer_name="char_rnn_encoding"):
    '''
        Char RNN Sequence Encoding Layer ..
    '''

    # representation learning layer
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        # GRU cell.
        gru = rnn.BasicLSTMCell(embed_dim, state_is_tuple=True)   # last dim size
        # drop_gru = gru
        drop_gru = tf.contrib.rnn.DropoutWrapper(gru, output_keep_prob=1.0 - dropout)
        encoder_cell = tf.contrib.rnn.MultiRNNCell([drop_gru] * 1, state_is_tuple=True)
        init_state = encoder_cell.zero_state(tf.shape(char_encoder_embed_input)[0], tf.float32)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, char_encoder_embed_input,
                                                           initial_state=init_state, dtype=tf.float32)
        encoder_all_outputs = encoder_outputs
        # Select last output.
        output = tf.transpose(encoder_outputs, [1, 0, 2])  # or time major =True
        char_encoder_state = tf.gather(output, int(output.get_shape()[0]) - 1)
    return char_encoder_state