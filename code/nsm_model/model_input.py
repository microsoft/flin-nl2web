import tensorflow as tf
from code.globals import max_no_dict, max_seq_len


def build_query_input_nodes(query_in):
    query_in['q_words'] = tf.placeholder(shape=[None, max_seq_len['query']], dtype=tf.int32,
                                           name='query_in')  # 32 x 15
    query_in['q_chars'] = tf.placeholder(shape=[None, max_seq_len['query'], max_seq_len['word']], dtype=tf.int32,
                                           name='query_in_char')  # 32 x 15 x 12
    query_in['q_ents'] = tf.placeholder(shape=[None, max_seq_len['query']], dtype=tf.int32,
                                          name='query_in_ent')  # 32 x 15

    query_in['q_para_val'] = tf.placeholder(shape=[None, max_seq_len['para_val']], dtype=tf.int32,
                                             name='q_para_val_in')  # 32 x 7
    query_in['q_para_val_char'] = tf.placeholder(shape=[None, max_seq_len['para_val'], max_seq_len['word']], dtype=tf.int32,
                                           name='q_para_val_char_in')  # 32 x 15 x 12

    query_in['tag_label'] = tf.placeholder(shape=[None, max_seq_len['query']], dtype=tf.int32,
                                         name='query_tag_in')  # 32 x 15
    query_in['q_len'] = tf.placeholder(shape=[None], dtype=tf.int32,
                                          name='query_len_in')  # 32 x 15

    query_in['bert_in_query'] = {'input_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_query']],
                                                            name='q_input_ids'),
                                'input_mask': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_query']],
                                                             name='q_input_mask'),
                                'segment_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_query']],
                                                              name='q_seg_ids')}

    query_in['bert_tagg_in'] = {'input_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_query_para_name']],
                                                             name='q_para_input_ids'),
                                 'input_mask': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_query_para_name']],
                                                              name='q_para_input_mask'),
                                 'segment_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_query_para_name']],
                                                               name='q_para_seg_ids')}

    query_in['bert_tag_label_st'] = tf.placeholder(shape=[None], dtype=tf.int64,
                                       name='start_id_in')  # 32 x 15
    query_in['bert_tag_label_end'] = tf.placeholder(shape=[None], dtype=tf.int64,
                                                   name='end_id_in')  # 32 x 15


def build_action_input_nodes(action_in):
    action_in['pos_act_name'] = tf.placeholder(shape=[None, max_seq_len['action_name']], dtype=tf.int32,
                                          name='pos_action_name_in')  # 32 x 5
    action_in['pos_act_para_names'] = tf.placeholder(shape=[None, max_no_dict['max_no_para_per_action'], max_seq_len['para_name']],
        dtype=tf.int32, name='pos_action_para_in')  # 32 x 10 x 5

    action_in['neg_act_name'] = tf.placeholder(shape=[None, max_seq_len['action_name']], dtype=tf.int32,
                                          name='neg_action_name_in')  # 32 x 5
    action_in['neg_act_para_names'] = tf.placeholder(shape=[None, max_no_dict['max_no_para_per_action'], max_seq_len['para_name']],
        dtype=tf.int32, name='neg_action_para_in')  # 32 x 10 x 5

    action_in['bert_in_pos_act_name'] = {'input_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_action_name']],
                                                    name='pos_act_name_input_ids'),
                                        'input_mask': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_action_name']],
                                                           name='pos_act_name_input_mask'),
                                        'segment_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_action_name']],
                                                           name='pos_act_name_seg_ids')}
    action_in['bert_in_pos_act_para_names'] = {'input_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_names_str']],
                                                     name='pos_act_para_names_input_ids'),
                                        'input_mask': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_names_str']],
                                                                     name='pos_act_para_names_input_mask'),
                                        'segment_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_names_str']],
                                                                      name='pos_act_para_names_seg_ids')}
    action_in['bert_in_neg_act_name'] = {'input_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_action_name']],
                                                       name='neg_act_name_input_ids'),
                                        'input_mask': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_action_name']],
                                                                     name='neg_act_name_input_mask'),
                                        'segment_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_action_name']],
                                                                      name='neg_act_name_seg_ids')}
    action_in['bert_in_neg_act_para_names'] = {'input_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_names_str']],
                                                      name='neg_act_para_names_input_ids'),
                                        'input_mask': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_names_str']],
                                                                     name='neg_act_para_names_input_mask'),
                                        'segment_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_names_str']],
                                                                      name='neg_act_para_names_seg_ids')}


def build_para_input_nodes(para_in):
    para_in['pos_para_name'] = tf.placeholder(shape=[None, max_seq_len['para_name']], dtype=tf.int32,
                                            name='pos_para_name_in')  # 32 x 5
    para_in['para_type'] = tf.placeholder(shape=[None], dtype=tf.float32,
                                          name='pos_para_type_in')  # 32

    # ==========================================
    para_in['pos_para_val'] = tf.placeholder(shape=[None, max_seq_len['para_val']], dtype=tf.int32,
                                           name='pos_para_val_in')  # 32 x 7
    para_in['pos_para_val_char'] = tf.placeholder(shape=[None, max_seq_len['para_val'], max_seq_len['word']],
            dtype=tf.int32, name='pos_val_in_char')  # 32 x 7 x 10

    para_in['neg_para_val'] = tf.placeholder(shape=[None, max_seq_len['para_val']], dtype=tf.int32,
                                           name='neg_para_val_in')  # 32 x 7
    para_in['neg_para_val_char'] = tf.placeholder(shape=[None, max_seq_len['para_val'], max_seq_len['word']],
            dtype=tf.int32, name='neg_val_in_char')  # 32 x 7 x 10

    para_in['q_para_match_pos'] = tf.placeholder(shape=[None, max_seq_len['query']], dtype=tf.int32,
                                                   name='query_para_match_pos') # 32 x 15
    para_in['q_para_match_neg'] = tf.placeholder(shape=[None, max_seq_len['query']], dtype=tf.int32,
                                                   name='query_para_match_neg')  # 32 x 15

    para_in['pos_ext_match_score'] = tf.placeholder(shape=[None, 1], dtype=tf.float32,
                                            name='pos_ext_match_score_in')  # 32 x 5
    para_in['neg_ext_match_score'] = tf.placeholder(shape=[None, 1], dtype=tf.float32,
                                                        name='neg_ext_match_score_in')  # 32 x 5

    para_in['bert_in_pos_para_name'] = {'input_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_name']],
                                                            name='para_name_input_ids'),
                                'input_mask': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_name']],
                                                             name='para_name_input_mask'),
                                'segment_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_name']],
                                                              name='para_name_seg_ids')}
    para_in['bert_in_pos_para_val'] = {'input_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_val']],
                                                           name='pos_para_val_input_ids'),
                                'input_mask': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_val']],
                                                             name='pos_para_val_input_mask'),
                                'segment_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_val']],
                                                              name='pos_para_val_seg_ids')}
    para_in['bert_in_neg_para_val'] = {'input_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_val']],
                                                             name='neg_para_val_input_ids'),
                                'input_mask': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_val']],
                                                             name='neg_para_val_input_mask'),
                                'segment_ids': tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len['bert_para_val']],
                                                              name='neg_para_val_seg_ids')}



