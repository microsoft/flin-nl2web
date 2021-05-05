import tensorflow_hub as hub

from code.nsm_model.action_embedding import get_action_embedding
from code.nsm_model.model_input import *
from code.nsm_model.para_val_semantics import encode_phrase, encode_phrase_char
from code.train_data_preprocessing.preprocess_util import char_vocab_to_id, ent_vocab_to_id
from code.nsm_model.Embedding_layer import *
from code.nsm_model.seq_learning import bert_encoding_layer
from code.nsm_model.Query_Encoding import encode_query
from tensorflow.contrib.layers.python.layers import initializers


def weight_variable(shape, name):
    return tf.get_variable(name+'_W', shape=shape,
                             dtype=tf.float32, initializer=initializers.xavier_initializer())


def bias_variable(out_dim, name):
    return tf.get_variable(name +'_b', shape=[out_dim],
                           dtype=tf.float32, initializer=tf.zeros_initializer())


class Neural_Semantic_Matching(object):

    def __init__(self, sess, vocab_size, embed_dim, lr, lamda, args):
        self.sess = sess

        self.class_dim = 2
        self.margin = 0.5
        self.embedding_dim = embed_dim
        self.embedding_dim_para = embed_dim
        self.learning_rate = lr
        self.enc_vocab_size = vocab_size
        self.char_vocab_size = len(char_vocab_to_id)
        self.ent_vocab_size = len(ent_vocab_to_id)
        self.lamda = lamda
        self.use_pre_train = False
        self.fine_tune = True
        self.args = args

        ''' Building the network model .. '''
        self.bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", trainable=True)

        self.read_input()
        self.build_para_tagging_network()
        self.get_all_embeddings()
        self.build_action_matching_network()
        self.build_para_matching_network()
        self.network_params = tf.trainable_variables()

        ''' All Model parameters ...'''
        self.variables_names = [v.name for v in self.network_params]

    def read_input(self):
        # input query
        self.query_in = {}
        build_query_input_nodes(self.query_in)

        # action inputs ...
        self.action_in = {}
        build_action_input_nodes(self.action_in)

        # para inputs ....
        self.para_in = {}
        build_para_input_nodes(self.para_in)

        # mode
        self.train_mode = tf.placeholder(tf.bool, name='train_mode')
        self.dropout = tf.cond(tf.equal(self.train_mode, True), lambda: 0.1, lambda: 0.0)

    def get_weights_act(self):
        return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.endswith('encoder_embedding_word:0')
                or v.name.endswith('rnn_encoding_act/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')
                or v.name.endswith('rnn_encoding_act/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')
                or v.name.endswith('projection_layer/weights:0')
                or v.name.endswith('p1_sim_weights:0')]

    def get_weights_para_tag(self):
        return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.endswith('start_W:0')
                or v.name.endswith('end_W:0')]

    def get_weights_para_match(self):
        return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.endswith('encoder_embedding_word:0')
                or v.name.endswith('encoder_embedding_char:0')
                or v.name.endswith('char_rnn_encoding/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0')
                or v.name.endswith('rnn_encoding_para2/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')
                or v.name.endswith('rnn_encoding_para2/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')
                or v.name.endswith('char_rnn_encoding_para2/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0')
                or v.name.endswith('rnn_encoding_para2_char2/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')
                or v.name.endswith('rnn_encoding_para2_char2/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')
                or v.name.endswith('fully_connected/weights:0')
                or v.name.endswith('lc_weights_closed:0')]

    def train_layer_act(self, reg_parameters):
        # loss function ..
        self.loss_act = tf.reduce_mean(tf.maximum(self.pos_dist_act - self.neg_dist_act + self.margin, 0))
        reg_loss_act = tf.reduce_mean([tf.nn.l2_loss(x) for x in reg_parameters])
        self.loss_act += 0.5 * self.lamda * reg_loss_act

        # learning rate decay
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate * 2.0)

        # compute gradients
        gradients_and_vars = optimizer.compute_gradients(self.loss_act)
        # Optimization Op
        self.optimize_act = optimizer.apply_gradients(gradients_and_vars, global_step=global_step)

    def train_layer_para_tag(self, reg_parameters):
        # loss function ..
        true_st = tf.one_hot(self.query_in['bert_tag_label_st'], max_seq_len['bert_query_para_name'])
        st_id_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.st_scores, labels=true_st)

        true_end = tf.one_hot(self.query_in['bert_tag_label_end'], max_seq_len['bert_query_para_name'])
        end_id_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.end_scores, labels=true_end)

        self.loss_para_tag = tf.reduce_mean(st_id_loss) + tf.reduce_mean(end_id_loss)

        reg_loss_para = tf.reduce_mean([tf.nn.l2_loss(x) for x in reg_parameters])
        self.loss_para_tag += 0.5 * self.lamda * reg_loss_para

        # learning rate decay
        global_step_p2 = tf.Variable(0, trainable=False)
        optimizer_p2 = tf.train.AdamOptimizer(self.learning_rate * 0.1)

        # compute gradients
        gradients_and_vars_p2 = optimizer_p2.compute_gradients(self.loss_para_tag)
        # Optimization Op
        self.optimize_para_tag = optimizer_p2.apply_gradients(gradients_and_vars_p2, global_step=global_step_p2)

    def train_layer_para_match(self, reg_parameters):
        # loss function ..
        self.loss_para = tf.reduce_mean(tf.maximum(self.pos_dist_para - self.neg_dist_para + self.margin, 0))
        self.reg_loss_para = tf.reduce_mean([tf.nn.l2_loss(x) for x in reg_parameters])
        self.loss_para += 0.5 * self.lamda * self.reg_loss_para

        # learning rate decay
        global_step_p2 = tf.Variable(0, trainable=False)
        optimizer_p2 = tf.train.AdamOptimizer(self.learning_rate * 2.0)

        # compute gradients
        gradients_and_vars_p2 = optimizer_p2.compute_gradients(self.loss_para)
        # Optimization Op
        self.optimize_para = optimizer_p2.apply_gradients(gradients_and_vars_p2, global_step=global_step_p2)

    def compute_cosine_dist(self, vec_a, vec_b):
        normalize_a = tf.nn.l2_normalize(vec_a, axis=1)   # 32 x 600
        normalize_b = tf.nn.l2_normalize(vec_b, axis=1)   # 32 x 600
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1)   # 32
        return tf.abs(tf.ones_like(cos_similarity) - cos_similarity) / 2.0   # 32

    def get_all_embeddings(self):
        # Embedding layers ....
        self.word_embed_encoder, self.embed_placeholder, self.embed_init = \
            word_embedding_layer(self.enc_vocab_size, self.embedding_dim, self.use_pre_train, is_trainable=self.fine_tune)

        self.char_embed_encoder = char_embedding_layer(self.char_vocab_size, self.embedding_dim_para, is_trainable=self.fine_tune)

        self.ent_embed_encoder = ent_embedding_layer(self.ent_vocab_size, self.embedding_dim_para * 2, is_trainable=self.fine_tune)

        self.match_embed_encoder = match_embedding_layer(self.embedding_dim_para * 2, True)

    def get_action_semantics(self, act_name, act_para_names, bert_act_name, bert_act_para_names, query_vec, query_states,
                             scope = "action_embedding_layer", scope2 = 'projection_layer'):
        # encoded pos action
        act_name_vec, act_para_vec = get_action_embedding(self.bert_module, self.word_embed_encoder, act_name, act_para_names,
                                                          bert_act_name, bert_act_para_names,
                                                          query_vec, query_states,
                                                          self.dropout, self.args, self.embedding_dim, scope)

        act_semantic_vec2 = tf.concat([act_name_vec, act_para_vec], axis=1)
        act_semantic_vec2 = tf.reshape(act_semantic_vec2, (tf.shape(act_semantic_vec2)[0], self.embedding_dim * 4))  # 32 x 1200
        act_semantic_vec = tf.contrib.layers.fully_connected(act_semantic_vec2, self.embedding_dim * 2,
                                                        activation_fn=tf.nn.tanh, scope=scope2, reuse=tf.AUTO_REUSE)  # 32 x 600
        return act_semantic_vec

    def build_action_matching_network(self):
        '''
            Action Intent Matching Network
        '''
        # encode query ..
        _, q_states, q_vec = encode_query(self, tag='action')

        # learn pos action semantics
        pos_act_semantic_vec = self.get_action_semantics(self.action_in['pos_act_name'],
                                                         self.action_in['pos_act_para_names'],
                                                         self.action_in['bert_in_pos_act_name'],
                                                         self.action_in['bert_in_pos_act_para_names'], q_vec, q_states)

        # learn neg action semantics
        neg_act_semantic_vec = self.get_action_semantics(self.action_in['neg_act_name'],
                                                         self.action_in['neg_act_para_names'],
                                                         self.action_in['bert_in_neg_act_name'],
                                                         self.action_in['bert_in_neg_act_para_names'], q_vec, q_states)

        # Scoring layer ...
        self.pos_dist_act = self.compute_cosine_dist(q_vec, pos_act_semantic_vec)
        self.neg_dist_act = self.compute_cosine_dist(q_vec, neg_act_semantic_vec)

        self.accuracy_act = tf.reduce_mean(tf.cast(tf.greater_equal(self.neg_dist_act, self.pos_dist_act), dtype=tf.float32))
        self.avg_pos_dist_act = tf.reduce_mean(self.pos_dist_act)
        self.avg_neg_dist_act = tf.reduce_mean(self.neg_dist_act)

        # training layer ..
        trainable_para_action = self.get_weights_act()
        self.train_layer_act(trainable_para_action)

    def build_para_tagging_network(self):
        '''
             Parameter Value Matching Network
        '''

        self.sequence_output, _ = bert_encoding_layer(self.bert_module, self.query_in['bert_tagg_in'], self.args)

        sequence_output_reshaped = tf.reshape(self.sequence_output, (tf.shape(self.sequence_output)[0]*
                                                                     max_seq_len['bert_query_para_name'], 768))

        self.start_H = weight_variable([768], 'start')
        self.end_H = weight_variable([768], 'end')

        st_scores = tf.reduce_sum(self.start_H * sequence_output_reshaped, axis=1)
        end_scores = tf.reduce_sum(self.end_H * sequence_output_reshaped, axis=1)

        self.st_scores = tf.reshape(st_scores, (tf.shape(self.sequence_output)[0], max_seq_len['bert_query_para_name']))
        self.end_scores = tf.reshape(end_scores, (tf.shape(self.sequence_output)[0], max_seq_len['bert_query_para_name']))

        self.pred_st_index = tf.argmax(self.st_scores, axis=1)
        self.pred_end_index = tf.argmax(self.end_scores, axis=1)

        ''' evaluation '''
        st_id_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred_st_index,
                                                self.query_in['bert_tag_label_st']),  dtype=tf.float32))
        end_id_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred_end_index,
                                                         self.query_in['bert_tag_label_end']), dtype=tf.float32))
        self.accuracy_para_tag = 0.5 * (st_id_accuracy + end_id_accuracy)

        # training layer ..
        trainable_para_p2 = self.get_weights_para_tag()
        self.train_layer_para_tag(trainable_para_p2)

    def build_para_matching_network(self):
        '''
             Parameter Value Matching Network
        '''

        ''' ================= Close domain Parameter matching : type 1  ======================= '''

        # Word Level ....
        para_name_vec = encode_phrase(self.bert_module, self.word_embed_encoder, self.para_in['pos_para_name'],
                                      self.para_in['bert_in_pos_para_name'],
                                      self.args, self.dropout, self.embedding_dim_para,
                                      'rnn_encoding_para2')  # 32 x 600

        q_para_val_vec = encode_phrase(self.bert_module, self.word_embed_encoder, self.query_in['q_para_val'],
                                      self.para_in['bert_in_pos_para_val'],
                                      self.args, self.dropout, self.embedding_dim_para, 'rnn_encoding_para2')  # 32 x 600
        pos_para_val_vec = encode_phrase(self.bert_module, self.word_embed_encoder, self.para_in['pos_para_val'],
                                         self.para_in['bert_in_pos_para_val'],
                                         self.args, self.dropout, self.embedding_dim_para, 'rnn_encoding_para2')  # 32 x 600
        neg_para_val_vec = encode_phrase(self.bert_module, self.word_embed_encoder, self.para_in['neg_para_val'],
                                         self.para_in['bert_in_neg_para_val'],
                                         self.args, self.dropout, self.embedding_dim_para, 'rnn_encoding_para2')  # 32 x 600

        pos_para_name_val_vec = tf.add(para_name_vec, pos_para_val_vec)
        neg_para_name_val_vec = tf.add(para_name_vec, neg_para_val_vec)

        # Scoring layer .....
        pos_dist_cls_learned = self.compute_cosine_dist(q_para_val_vec, pos_para_name_val_vec)
        neg_dist_cls_learned = self.compute_cosine_dist(q_para_val_vec, neg_para_name_val_vec)

        pos_dist_cls_learned = tf.reshape(pos_dist_cls_learned, (tf.shape(pos_dist_cls_learned)[0], 1))
        neg_dist_cls_learned = tf.reshape(neg_dist_cls_learned, (tf.shape(neg_dist_cls_learned)[0], 1))

        # CHAR LEVEL....
        q_para_val_vec_char = encode_phrase_char(self.bert_module, self.char_embed_encoder, self.query_in['q_para_val_char'],
                                       max_seq_len['para_val'], self.para_in['bert_in_pos_para_val'],
                                       self.args, self.dropout, self.embedding_dim_para,
                                       'rnn_encoding_para2')  # 32 x 600
        pos_para_val_vec_char = encode_phrase_char(self.bert_module, self.char_embed_encoder, self.para_in['pos_para_val_char'],
                                       max_seq_len['para_val'], self.para_in['bert_in_pos_para_val'],
                                       self.args, self.dropout, self.embedding_dim_para,
                                       'rnn_encoding_para2')  # 32 x 600
        neg_para_val_vec_char = encode_phrase_char(self.bert_module, self.char_embed_encoder, self.para_in['neg_para_val_char'],
                                       max_seq_len['para_val'], self.para_in['bert_in_pos_para_val'],
                                       self.args, self.dropout, self.embedding_dim_para,
                                       'rnn_encoding_para2')  # 32 x 600

        pos_para_name_val_vec_char = tf.add(para_name_vec, pos_para_val_vec_char)
        neg_para_name_val_vec_char = tf.add(para_name_vec, neg_para_val_vec_char)

        # Scoring layer .....
        pos_dist_cls_char = self.compute_cosine_dist(q_para_val_vec_char, pos_para_name_val_vec_char)
        neg_dist_cls_char = self.compute_cosine_dist(q_para_val_vec_char, neg_para_name_val_vec_char)

        pos_dist_cls_char = tf.reshape(pos_dist_cls_char, (tf.shape(pos_dist_cls_char)[0], 1))
        neg_dist_cls_char = tf.reshape(neg_dist_cls_char, (tf.shape(neg_dist_cls_char)[0], 1))

        if self.args['model_result_dict'] == 'flin-sem':
            print ('in FLIN SEM')
            pos_dist_closed = tf.concat([pos_dist_cls_learned, pos_dist_cls_char],  axis=1)  # 32 x 2
            neg_dist_closed = tf.concat([neg_dist_cls_learned, neg_dist_cls_char],  axis=1)  # 32 x 2
        elif self.args['model_result_dict'] == 'flin-lex':
            print ('in FLIN LEX')
            pos_dist_closed = self.para_in['pos_ext_match_score']
            neg_dist_closed = self.para_in['neg_ext_match_score']
        else:
            pos_dist_closed = tf.concat([pos_dist_cls_learned, pos_dist_cls_char,
                                         self.para_in['pos_ext_match_score']], axis=1)  # 32 x 3
            neg_dist_closed = tf.concat([neg_dist_cls_learned, neg_dist_cls_char,
                                         self.para_in['neg_ext_match_score']], axis=1)  # 32 x 3

        self.pos_dist_para = tf.reduce_mean(pos_dist_closed, axis=1)
        self.neg_dist_para = tf.reduce_mean(neg_dist_closed, axis=1)

        ''' evaluation '''
        self.accuracy_para = tf.reduce_mean(tf.cast(tf.greater_equal(self.neg_dist_para, self.pos_dist_para), dtype=tf.float32))
        self.avg_pos_dist_para = tf.reduce_mean(self.pos_dist_para)
        self.avg_neg_dist_para = tf.reduce_mean(self.neg_dist_para)

        # training layer ..
        trainable_para_p2 = self.get_weights_para_match()
        self.train_layer_para_match(trainable_para_p2)

    def train_action(self, batch_action):
        return self.sess.run([self.optimize_act, self.loss_act, self.avg_pos_dist_act,
                              self.avg_neg_dist_act, self.accuracy_act], feed_dict={
            self.query_in['q_words']: batch_action['query'],
            self.action_in['pos_act_name']: batch_action['pos_act_name'],
            self.action_in['pos_act_para_names']: batch_action['pos_act_para_names'],
            self.action_in['neg_act_name']: batch_action['neg_act_name'],
            self.action_in['neg_act_para_names']: batch_action['neg_act_para_names'],

            self.query_in['bert_in_query']['input_ids']: batch_action['bert_in_query'][0],
            self.query_in['bert_in_query']['input_mask']: batch_action['bert_in_query'][1],
            self.query_in['bert_in_query']['segment_ids']: batch_action['bert_in_query'][2],

            self.action_in['bert_in_pos_act_name']['input_ids']: batch_action['bert_in_pos_act_name'][0],
            self.action_in['bert_in_pos_act_name']['input_mask']: batch_action['bert_in_pos_act_name'][1],
            self.action_in['bert_in_pos_act_name']['segment_ids']: batch_action['bert_in_pos_act_name'][2],
            self.action_in['bert_in_pos_act_para_names']['input_ids']: batch_action['bert_in_pos_act_para_names'][0],
            self.action_in['bert_in_pos_act_para_names']['input_mask']: batch_action['bert_in_pos_act_para_names'][1],
            self.action_in['bert_in_pos_act_para_names']['segment_ids']: batch_action['bert_in_pos_act_para_names'][2],

            self.action_in['bert_in_neg_act_name']['input_ids']: batch_action['bert_in_neg_act_name'][0],
            self.action_in['bert_in_neg_act_name']['input_mask']: batch_action['bert_in_neg_act_name'][1],
            self.action_in['bert_in_neg_act_name']['segment_ids']: batch_action['bert_in_neg_act_name'][2],
            self.action_in['bert_in_neg_act_para_names']['input_ids']: batch_action['bert_in_neg_act_para_names'][0],
            self.action_in['bert_in_neg_act_para_names']['input_mask']: batch_action['bert_in_neg_act_para_names'][1],
            self.action_in['bert_in_neg_act_para_names']['segment_ids']: batch_action['bert_in_neg_act_para_names'][2],

            self.train_mode: True
        })

    def evaluate_action(self, batch_action):
        return self.sess.run([self.loss_act, self.avg_pos_dist_act, self.avg_neg_dist_act, self.accuracy_act], feed_dict={
            self.query_in['q_words']: batch_action['query'],
            self.action_in['pos_act_name']: batch_action['pos_act_name'],
            self.action_in['pos_act_para_names']: batch_action['pos_act_para_names'],
            self.action_in['neg_act_name']: batch_action['neg_act_name'],
            self.action_in['neg_act_para_names']: batch_action['neg_act_para_names'],

            self.query_in['bert_in_query']['input_ids']: batch_action['bert_in_query'][0],
            self.query_in['bert_in_query']['input_mask']: batch_action['bert_in_query'][1],
            self.query_in['bert_in_query']['segment_ids']: batch_action['bert_in_query'][2],

            self.action_in['bert_in_pos_act_name']['input_ids']: batch_action['bert_in_pos_act_name'][0],
            self.action_in['bert_in_pos_act_name']['input_mask']: batch_action['bert_in_pos_act_name'][1],
            self.action_in['bert_in_pos_act_name']['segment_ids']: batch_action['bert_in_pos_act_name'][2],
            self.action_in['bert_in_pos_act_para_names']['input_ids']: batch_action['bert_in_pos_act_para_names'][0],
            self.action_in['bert_in_pos_act_para_names']['input_mask']: batch_action['bert_in_pos_act_para_names'][1],
            self.action_in['bert_in_pos_act_para_names']['segment_ids']: batch_action['bert_in_pos_act_para_names'][2],

            self.action_in['bert_in_neg_act_name']['input_ids']: batch_action['bert_in_neg_act_name'][0],
            self.action_in['bert_in_neg_act_name']['input_mask']: batch_action['bert_in_neg_act_name'][1],
            self.action_in['bert_in_neg_act_name']['segment_ids']: batch_action['bert_in_neg_act_name'][2],
            self.action_in['bert_in_neg_act_para_names']['input_ids']: batch_action['bert_in_neg_act_para_names'][0],
            self.action_in['bert_in_neg_act_para_names']['input_mask']: batch_action['bert_in_neg_act_para_names'][1],
            self.action_in['bert_in_neg_act_para_names']['segment_ids']: batch_action['bert_in_neg_act_para_names'][2],
            self.train_mode: False
        })

    def predict_action(self, batch_action):
        return self.sess.run(self.pos_dist_act, feed_dict={
            self.query_in['q_words']: batch_action['query'],
            self.action_in['pos_act_name']: batch_action['pos_act_name'],
            self.action_in['pos_act_para_names']: batch_action['pos_act_para_names'],

            self.query_in['bert_in_query']['input_ids']: batch_action['bert_in_query'][0],
            self.query_in['bert_in_query']['input_mask']: batch_action['bert_in_query'][1],
            self.query_in['bert_in_query']['segment_ids']: batch_action['bert_in_query'][2],

            self.action_in['bert_in_pos_act_name']['input_ids']: batch_action['bert_in_pos_act_name'][0],
            self.action_in['bert_in_pos_act_name']['input_mask']: batch_action['bert_in_pos_act_name'][1],
            self.action_in['bert_in_pos_act_name']['segment_ids']: batch_action['bert_in_pos_act_name'][2],
            self.action_in['bert_in_pos_act_para_names']['input_ids']: batch_action['bert_in_pos_act_para_names'][0],
            self.action_in['bert_in_pos_act_para_names']['input_mask']: batch_action['bert_in_pos_act_para_names'][1],
            self.action_in['bert_in_pos_act_para_names']['segment_ids']: batch_action['bert_in_pos_act_para_names'][2],

            self.train_mode: False
        })

    def train_para_tag(self, batch_para):
        return self.sess.run([self.optimize_para_tag, self.loss_para_tag, self.accuracy_para_tag], feed_dict={

            self.query_in['bert_tagg_in']['input_ids']: batch_para['bert_tagg_in'][0],
            self.query_in['bert_tagg_in']['input_mask']: batch_para['bert_tagg_in'][1],
            self.query_in['bert_tagg_in']['segment_ids']: batch_para['bert_tagg_in'][2],

            self.query_in['bert_tag_label_st']: batch_para['bert_tag_label_st'],
            self.query_in['bert_tag_label_end']: batch_para['bert_tag_label_end'],

            self.train_mode: True
        })

    def evaluate_para_tag(self, batch_para):
        return self.sess.run([self.loss_para_tag, self.accuracy_para_tag,
                              self.sequence_output], feed_dict={
            self.query_in['bert_tagg_in']['input_ids']: batch_para['bert_tagg_in'][0],
            self.query_in['bert_tagg_in']['input_mask']: batch_para['bert_tagg_in'][1],
            self.query_in['bert_tagg_in']['segment_ids']: batch_para['bert_tagg_in'][2],

            self.query_in['bert_tag_label_st']: batch_para['bert_tag_label_st'],
            self.query_in['bert_tag_label_end']: batch_para['bert_tag_label_end'],
            self.train_mode: False
        })

    def predict_para_tag(self, batch_para):
        return self.sess.run([self.pred_st_index, self.pred_end_index], feed_dict={
            self.query_in['bert_tagg_in']['input_ids']: batch_para['bert_tagg_in'][0],
            self.query_in['bert_tagg_in']['input_mask']: batch_para['bert_tagg_in'][1],
            self.query_in['bert_tagg_in']['segment_ids']: batch_para['bert_tagg_in'][2],

            self.train_mode: False
        })

    def train_para_match(self, batch_para):
        return self.sess.run([self.optimize_para, self.loss_para,
                              self.avg_pos_dist_para, self.avg_neg_dist_para, self.accuracy_para], feed_dict={
            self.query_in['q_para_val']: batch_para['q_para_val'],
            self.query_in['q_para_val_char']: batch_para['q_para_val_char'],

            self.para_in['para_type']: batch_para['para_type'],
            self.para_in['pos_para_name']: batch_para['pos_para_name'],

            self.para_in['pos_para_val']: batch_para['pos_para_val'],
            self.para_in['pos_para_val_char']: batch_para['pos_para_val_char'],
            self.para_in['pos_ext_match_score']: batch_para['pos_ext_match_score'],

            self.para_in['neg_para_val']: batch_para['neg_para_val'],
            self.para_in['neg_para_val_char']: batch_para['neg_para_val_char'],
            self.para_in['neg_ext_match_score']: batch_para['neg_ext_match_score'],

            self.para_in['bert_in_pos_para_name']['input_ids']: batch_para['bert_in_pos_para_name'][0],
            self.para_in['bert_in_pos_para_name']['input_mask']: batch_para['bert_in_pos_para_name'][1],
            self.para_in['bert_in_pos_para_name']['segment_ids']: batch_para['bert_in_pos_para_name'][2],

            self.para_in['bert_in_pos_para_val']['input_ids']: batch_para['bert_in_pos_para_val'][0],
            self.para_in['bert_in_pos_para_val']['input_mask']: batch_para['bert_in_pos_para_val'][1],
            self.para_in['bert_in_pos_para_val']['segment_ids']: batch_para['bert_in_pos_para_val'][2],

            self.para_in['bert_in_neg_para_val']['input_ids']: batch_para['bert_in_neg_para_val'][0],
            self.para_in['bert_in_neg_para_val']['input_mask']: batch_para['bert_in_neg_para_val'][1],
            self.para_in['bert_in_neg_para_val']['segment_ids']: batch_para['bert_in_neg_para_val'][2],

            self.train_mode: True
        })

    def evaluate_para_match(self, batch_para):
        return self.sess.run([self.loss_para, self.avg_pos_dist_para,
                              self.avg_neg_dist_para, self.accuracy_para], feed_dict={
            self.query_in['q_para_val']: batch_para['q_para_val'],
            self.query_in['q_para_val_char']: batch_para['q_para_val_char'],

            self.para_in['para_type']: batch_para['para_type'],
            self.para_in['pos_para_name']: batch_para['pos_para_name'],

            self.para_in['pos_para_val']: batch_para['pos_para_val'],
            self.para_in['pos_para_val_char']: batch_para['pos_para_val_char'],
            self.para_in['pos_ext_match_score']: batch_para['pos_ext_match_score'],

            self.para_in['neg_para_val']: batch_para['neg_para_val'],
            self.para_in['neg_para_val_char']: batch_para['neg_para_val_char'],
            self.para_in['neg_ext_match_score']: batch_para['neg_ext_match_score'],

            self.para_in['bert_in_pos_para_name']['input_ids']: batch_para['bert_in_pos_para_name'][0],
            self.para_in['bert_in_pos_para_name']['input_mask']: batch_para['bert_in_pos_para_name'][1],
            self.para_in['bert_in_pos_para_name']['segment_ids']: batch_para['bert_in_pos_para_name'][2],

            self.para_in['bert_in_pos_para_val']['input_ids']: batch_para['bert_in_pos_para_val'][0],
            self.para_in['bert_in_pos_para_val']['input_mask']: batch_para['bert_in_pos_para_val'][1],
            self.para_in['bert_in_pos_para_val']['segment_ids']: batch_para['bert_in_pos_para_val'][2],

            self.para_in['bert_in_neg_para_val']['input_ids']: batch_para['bert_in_neg_para_val'][0],
            self.para_in['bert_in_neg_para_val']['input_mask']: batch_para['bert_in_neg_para_val'][1],
            self.para_in['bert_in_neg_para_val']['segment_ids']: batch_para['bert_in_neg_para_val'][2],
            self.train_mode: False
        })

    def predict_para_match(self, batch_para):
        return self.sess.run(self.pos_dist_para, feed_dict={
            self.query_in['q_para_val']: batch_para['q_para_val'],
            self.query_in['q_para_val_char']: batch_para['q_para_val_char'],

            self.para_in['para_type']: batch_para['para_type'],
            self.para_in['pos_para_name']: batch_para['pos_para_name'],

            self.para_in['pos_para_val']: batch_para['pos_para_val'],
            self.para_in['pos_para_val_char']: batch_para['pos_para_val_char'],
            self.para_in['pos_ext_match_score']: batch_para['pos_ext_match_score'],

            self.para_in['bert_in_pos_para_name']['input_ids']: batch_para['bert_in_pos_para_name'][0],
            self.para_in['bert_in_pos_para_name']['input_mask']: batch_para['bert_in_pos_para_name'][1],
            self.para_in['bert_in_pos_para_name']['segment_ids']: batch_para['bert_in_pos_para_name'][2],

            self.para_in['bert_in_pos_para_val']['input_ids']: batch_para['bert_in_pos_para_val'][0],
            self.para_in['bert_in_pos_para_val']['input_mask']: batch_para['bert_in_pos_para_val'][1],
            self.para_in['bert_in_pos_para_val']['segment_ids']: batch_para['bert_in_pos_para_val'][2],

            self.train_mode: False
        })

    def initialize_embedding(self, embedding):
        self.sess.run(self.embed_init, feed_dict={
            self.embed_placeholder: embedding
        })

    def print_variable_names(self):
        values = self.sess.run(self.variables_names)

        for k, v in zip(self.variables_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print("================\n")