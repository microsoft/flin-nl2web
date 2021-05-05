import os
import random

import numpy as np
import tensorflow as tf

random.seed(1234)


def load_or_initialize_model(sess, saver, model_name, model_path):
    sess.run(tf.global_variables_initializer())

    if os.path.isfile(model_path+model_name+'/'+model_name+'.ckpt.meta'):
        saver.restore(sess, model_path+model_name+'/'+model_name+'.ckpt')
        print(model_name+" Model restored.")
        return True
    else:
        print(model_name+ " Model initialized.")
        return False


def get_shuffled_train_data(data):
    for k in range(10):
        random.shuffle(data)
    return data


def save_model(sess, saver, model_name, model_path):
    # Save model weights to disk

    if not os.path.isdir(model_path + model_name + '/'):
        os.makedirs(model_path + model_name + '/')

    save_path = saver.save(sess, model_path + model_name + '/' + model_name+".ckpt")
    print(" Model saved in file: %s at episode:" % save_path)


def compute_taging_accuracy(batch_X_tag_label, batch_X_q_len, pred_viterbi_sequence):
    acc = []
    for i in range(len(batch_X_tag_label)):
        label_vec = batch_X_tag_label[i][:batch_X_q_len[i]]
        pred_sequence_vec = pred_viterbi_sequence[i][:batch_X_q_len[i]]

        acc.append(np.mean(pred_sequence_vec[label_vec == 1]))
    return np.mean(acc)


def get_next_batch_action(data, itr, batch_size):

    batch = data[(itr * batch_size):(itr * batch_size) + batch_size]

    # categorical_input
    batch_X_q = []

    batch_X_pos_act_name = []
    batch_X_pos_act_para_names = []

    batch_X_neg_act_name = []
    batch_X_neg_act_para_names = []

    bert_X_query = { 'input_ids': [], 'input_mask': [], 'segment_ids':[]}
    bert_X_pos_action_name = {'input_ids': [], 'input_mask': [], 'segment_ids': []}
    bert_X_pos_act_para_names = {'input_ids': [], 'input_mask': [], 'segment_ids': []}

    bert_X_neg_action_name = {'input_ids': [], 'input_mask': [], 'segment_ids': []}
    bert_X_neg_act_para_names = {'input_ids': [], 'input_mask': [], 'segment_ids': []}

    for q_vec, pos_action_name, pos_action_para_names, pos_action_para_dom, \
                    neg_act_list, pos_act_bert_input, neg_act_bert_input_list in batch:

        batch_X_q.append(q_vec)

        batch_X_pos_act_name.append(pos_action_name)
        batch_X_pos_act_para_names.append(pos_action_para_names)

        neg_act_tup = random.choice(neg_act_list)

        batch_X_neg_act_name.append(neg_act_tup[0])
        batch_X_neg_act_para_names.append(neg_act_tup[1])

        # bert inputs
        add_bert_input(bert_X_query, pos_act_bert_input[0])
        add_bert_input(bert_X_pos_action_name, pos_act_bert_input[1])
        add_bert_input(bert_X_pos_act_para_names, pos_act_bert_input[2])

        neg_bert_input_p1 = random.choice(neg_act_bert_input_list)
        add_bert_input(bert_X_neg_action_name, neg_bert_input_p1[0])
        add_bert_input(bert_X_neg_act_para_names, neg_bert_input_p1[1])

    action_batch = {'query': np.array(batch_X_q), 'pos_act_name': np.array(batch_X_pos_act_name),
                    'pos_act_para_names': np.array(batch_X_pos_act_para_names),
                    'neg_act_name': np.array(batch_X_neg_act_name), 'neg_act_para_names': np.array(batch_X_neg_act_para_names),

                    'bert_in_query': (np.array(bert_X_query['input_ids']),
                                                np.array(bert_X_query['input_mask']),
                                                np.array(bert_X_query['segment_ids'])),

                    'bert_in_pos_act_name': (np.array(bert_X_pos_action_name['input_ids']),
                                                np.array(bert_X_pos_action_name['input_mask']),
                                                np.array(bert_X_pos_action_name['segment_ids'])),
                    'bert_in_pos_act_para_names': (np.array(bert_X_pos_act_para_names['input_ids']),
                                                   np.array(bert_X_pos_act_para_names['input_mask']),
                                                   np.array(bert_X_pos_act_para_names['segment_ids'])),

                    'bert_in_neg_act_name': (np.array(bert_X_neg_action_name['input_ids']),
                                                np.array(bert_X_neg_action_name['input_mask']),
                                                np.array(bert_X_neg_action_name['segment_ids'])),
                    'bert_in_neg_act_para_names': (np.array(bert_X_neg_act_para_names['input_ids']),
                                                   np.array(bert_X_neg_act_para_names['input_mask']),
                                                   np.array(bert_X_neg_act_para_names['segment_ids']))

                    }

    return action_batch


def add_bert_input(batch_X_bert, bert_in_tup):
    batch_X_bert['input_ids'].append(bert_in_tup[0])
    batch_X_bert['input_mask'].append(bert_in_tup[1])
    batch_X_bert['segment_ids'].append(bert_in_tup[2])


def get_next_batch_para_tagger(data, itr, batch_size):

    batch = data[(itr * batch_size):(itr * batch_size) + batch_size]

    # categorical_input
    batch_X_q = []
    batch_X_q_char = []
    batch_X_q_ent = []

    batch_X_pos_para_name = []
    batch_X_para_type = []

    batch_X_label_seq = []
    batch_X_q_len = []

    bert_X_tagg_in = {'input_ids': [], 'input_mask': [], 'segment_ids': []}
    bert_X_label_st = []
    bert_X_label_end = []

    for q_vec, pos_para_type, pos_para_name, label_vec, q_len, \
               q_char_vec, q_ent_vec, bert_input_tagging, gold_label_ids in batch:

        batch_X_q.append(q_vec)
        batch_X_q_char.append(q_char_vec)
        batch_X_q_ent.append(q_ent_vec)

        batch_X_pos_para_name.append(pos_para_name)
        batch_X_para_type.append(pos_para_type)

        batch_X_label_seq.append(label_vec)
        batch_X_q_len.append(q_len)

        # bert inputs
        add_bert_input(bert_X_tagg_in, bert_input_tagging)

        bert_X_label_st.append(gold_label_ids[0])
        bert_X_label_end.append(gold_label_ids[1])

    para_batch = {'query': np.array(batch_X_q), 'q_char': np.array(batch_X_q_char), 'q_ent': np.array(batch_X_q_ent),
                  'para_type': np.array(batch_X_para_type),
                  'pos_para_name': np.array(batch_X_pos_para_name),

                  'tag_label': np.array(batch_X_label_seq),
                  'q_len': np.array(batch_X_q_len),

                  'bert_tagg_in': (np.array(bert_X_tagg_in['input_ids']),
                                    np.array(bert_X_tagg_in['input_mask']),
                                    np.array(bert_X_tagg_in['segment_ids'])),

                  'bert_tag_label_st': np.array(bert_X_label_st),
                  'bert_tag_label_end': np.array(bert_X_label_end)
                 }

    return para_batch


def get_next_batch_para_matcher(data, itr, batch_size):

    batch = data[(itr * batch_size):(itr * batch_size) + batch_size]

    batch_X_q_para_val = []
    batch_X_q_para_val_char = []

    batch_X_pos_para_name = []
    batch_X_para_type = []
    batch_X_pos_para_val = []
    batch_X_pos_para_val_char = []
    batch_X_pos_ext_match_score = []

    batch_X_neg_para_val = []
    batch_X_neg_para_val_char = []
    batch_X_neg_ext_match_score = []

    bert_X_pos_para_name = {'input_ids': [], 'input_mask': [], 'segment_ids': []}
    bert_X_pos_para_val = {'input_ids': [], 'input_mask': [], 'segment_ids': []}
    bert_X_neg_para_val = {'input_ids': [], 'input_mask': [], 'segment_ids': []}

    for q_para_val_vec, pos_para_type, pos_para_name, pos_para_val, pos_ext_match_score, \
                neg_para_val_list, pos_para_bert_input, neg_para_bert_input_list, \
                q_para_val_vec_char, pos_para_val_vec_char in batch:

        batch_X_q_para_val.append(q_para_val_vec)
        batch_X_q_para_val_char.append(q_para_val_vec_char)

        batch_X_pos_para_name.append(pos_para_name)
        batch_X_para_type.append(pos_para_type)
        batch_X_pos_para_val.append(pos_para_val)
        batch_X_pos_para_val_char.append(pos_para_val_vec_char)

        batch_X_pos_ext_match_score.append(pos_ext_match_score)

        neg_tup = random.choice(neg_para_val_list)   # (neg_para_val_vec, neg_val_ext_match_score,
                                                             # neg_para_val_vec_char, q_para_match_vec_neg)
        batch_X_neg_para_val.append(neg_tup[0])
        batch_X_neg_ext_match_score.append(neg_tup[1])
        batch_X_neg_para_val_char.append(neg_tup[2])

        # bert inputs
        add_bert_input(bert_X_pos_para_name, pos_para_bert_input[0])
        add_bert_input(bert_X_pos_para_val, pos_para_bert_input[1])

        neg_bert_input_p1 = random.choice(neg_para_bert_input_list)
        add_bert_input(bert_X_neg_para_val, neg_bert_input_p1)

    para_batch = {'q_para_val': np.array(batch_X_q_para_val),
                  'q_para_val_char': np.array(batch_X_q_para_val_char),

                  'para_type': np.array(batch_X_para_type),
                  'pos_para_name': np.array(batch_X_pos_para_name),

                  'pos_para_val': np.array(batch_X_pos_para_val),
                  'pos_para_val_char': np.array(batch_X_pos_para_val_char),
                  'pos_ext_match_score': np.array(batch_X_pos_ext_match_score),

                  'neg_para_val': np.array(batch_X_neg_para_val),
                  'neg_para_val_char': np.array(batch_X_neg_para_val_char),
                  'neg_ext_match_score': np.array(batch_X_neg_ext_match_score),

                  'bert_in_pos_para_name': (np.array(bert_X_pos_para_name['input_ids']),
                                              np.array(bert_X_pos_para_name['input_mask']),
                                              np.array(bert_X_pos_para_name['segment_ids'])),

                  'bert_in_pos_para_val': (np.array(bert_X_pos_para_val['input_ids']),
                                                 np.array(bert_X_pos_para_val['input_mask']),
                                                 np.array(bert_X_pos_para_val['segment_ids'])),
                  'bert_in_neg_para_val': (np.array(bert_X_neg_para_val['input_ids']),
                                              np.array(bert_X_neg_para_val['input_mask']),
                                              np.array(bert_X_neg_para_val['segment_ids'])),
                 }

    return para_batch
