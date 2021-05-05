from code.nsm_model.seq_learning import *


def encode_query(model, tag):
    # encode query ....
    encoded_query_states = None
    encoded_query_vec = None
    mp_embedding = None

    if tag == 'action':
        q_word_emb_inputs = tf.nn.embedding_lookup(model.word_embed_encoder, model.query_in['q_words'])
        mp_embedding = q_word_emb_inputs  # ONLY WORD

        # final query encoding ...
        if model.args['encoder_type'] == 'rnn_bi' or model.args['pred_model_type'] == 'rnn_uni':
            encoded_query_states, encoded_query_vec = rnn_encoding_layer(model.dropout, mp_embedding, model.args,
                                                                            model.embedding_dim_para, 'rnn_encoding_act')
        elif model.args['encoder_type'] == 'bert':
            encoded_query_states, encoded_query_vec = bert_encoding_layer(model.bert_module,
                                                                          model.query_in['bert_in_query'], model.args)

    return mp_embedding, encoded_query_states, encoded_query_vec
