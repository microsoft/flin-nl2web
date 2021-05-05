from code.globals import max_no_dict, max_seq_len
from code.nsm_model.para_val_semantics import encode_phrase
from code.nsm_model.seq_learning import *


def get_action_embedding(bert_module, word_embed_encoder, act_name, act_para_names, bert_act_name, bert_act_para_names,
                         query_vec, query_states, dropout, args, embed_dim, scope):
    '''
     parameter domain or examples are excluded for now from learning ...
    :param action_name:
    :param parameters:
    :return:
    '''

    # action name encoding ...
    action_name_vec = encode_action_name(bert_module, word_embed_encoder, act_name, bert_act_name,
                                         query_vec, query_states,  args, dropout, embed_dim, scope)

    # encode action parameter names  ...
    if args['encoder_type'] == 'rnn_bi' or args['encoder_type'] == 'rnn_uni'  or args['encoder_type'] == 'cnn':
        action_para_name_embs = encode_action_para_names(word_embed_encoder, act_para_names, args, dropout, embed_dim)    # 32 x 10 x 600
        action_para_name_vec = tf.reduce_mean(action_para_name_embs, axis=1)    # 32 x 600
    else:
        _, action_para_name_vec = encode_phrase(bert_module, word_embed_encoder, act_para_names, bert_act_para_names,
                                                args, dropout, embed_dim, 'rnn_encoding_act')

    return action_name_vec, action_para_name_vec


def encode_action_name(bert_module, word_embed_encoder, action_name_in, bert_action_name_in,
                       query_vec, query_states,  args, dropout, embed_dim, scope):

    # encode action name ...
    if args['encoder_type'] == 'rnn_bi' or args['encoder_type'] == 'rnn_uni':
        action_name_emb_inputs = tf.nn.embedding_lookup(word_embed_encoder, action_name_in)
        _, action_name_vec = rnn_encoding_layer(dropout, action_name_emb_inputs, args,
                                             embed_dim, 'rnn_encoding_act')  # 32 x 5 x 300 ---> 32 x 600
    elif args['encoder_type'] == 'cnn':
        action_name_emb_inputs = tf.nn.embedding_lookup(word_embed_encoder, action_name_in)
        action_name_vec = cnn_encoding_layer(dropout, action_name_emb_inputs, args, embed_dim,
                                             max_seq_len['action_name'])  # 32 x 5 x 300 ---> 32 x 600
    else:
        _, action_name_vec = bert_encoding_layer(bert_module, bert_action_name_in, args)

    return action_name_vec


def encode_action_para_names(word_embed_encoder, action_para_in, args, dropout, embed_dim):
    action_para_embs_inputs = tf.nn.embedding_lookup(word_embed_encoder, action_para_in)  # 32 x 10 x 5 x 300
    action_para_embs_inputs_2 = tf.reshape(action_para_embs_inputs,
                                    (tf.shape(action_para_embs_inputs)[0] * max_no_dict['max_no_para_per_action'],
                                     max_seq_len['para_name'], embed_dim))

    if args['encoder_type'] == 'rnn_bi' or args['encoder_type'] == 'rnn_uni':
        _, action_para_embs = rnn_encoding_layer(dropout, action_para_embs_inputs_2,
                                                 args, embed_dim, 'rnn_encoding_act')
    else:
        action_para_embs = cnn_encoding_layer(dropout, action_para_embs_inputs_2, args, embed_dim,
                                              max_seq_len['para_name'])

    action_para_name_embs = tf.reshape(action_para_embs, (tf.shape(action_para_embs_inputs)[0],
                                                     max_no_dict['max_no_para_per_action'],
                                                     tf.shape(action_para_embs)[1]))  # 32 x 10 x 600
    return action_para_name_embs
