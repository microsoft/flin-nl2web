from code.globals import max_seq_len
from code.nsm_model.Embedding_layer import get_char_embeddings_phrase
from code.nsm_model.seq_learning import *


def encode_phrase_char(bert_module, char_embed_encoder, input_seq, seq_len, bert_input, args, dropout, embed_dim, layer_name):
    phrase_char_emb_inputs = get_char_embeddings_phrase(dropout, input_seq, char_embed_encoder,
                                    seq_len, max_seq_len['word'], embed_dim, args, 'char_rnn_encoding_para2')

    if args['encoder_type'] == 'rnn_bi' or args['pred_model_type'] == 'rnn_uni':
        _, phrase_vec = rnn_encoding_layer(dropout, phrase_char_emb_inputs, args, embed_dim, layer_name+'_char2')
    elif args['encoder_type'] == 'cnn':
        phrase_vec = cnn_encoding_layer(dropout, phrase_char_emb_inputs, args, embed_dim, max_seq_len['para_name'])
    else:
        _, phrase_vec = bert_encoding_layer(bert_module, bert_input, args)

    return phrase_vec


def encode_phrase(bert_module, word_embed_encoder, input_seq, bert_input, args, dropout, embed_dim, layer_name):
    if args['encoder_type'] == 'rnn_bi' or args['pred_model_type'] == 'rnn_uni':
        phrase_embs_inputs = tf.nn.embedding_lookup(word_embed_encoder, input_seq)  # 32 x 5 x 300
        _, phrase_vec = rnn_encoding_layer(dropout, phrase_embs_inputs, args, embed_dim, layer_name)
    elif args['encoder_type'] == 'cnn':
        phrase_embs_inputs = tf.nn.embedding_lookup(word_embed_encoder, input_seq)  # 32 x 5 x 300
        phrase_vec = cnn_encoding_layer(dropout, phrase_embs_inputs, args, embed_dim, max_seq_len['para_name'])
    else:
        _, phrase_vec = bert_encoding_layer(bert_module, bert_input, args)

    return phrase_vec
