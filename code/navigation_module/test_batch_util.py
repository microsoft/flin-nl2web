from code.navigation_module.util_navigation import *
from code.nsm_model.util_helper import add_bert_input


def get_next_test_batch_action(data, itr, batch_size):

    batch = data[(itr * batch_size):(itr * batch_size) + batch_size]

    # categorical_input
    batch_X_q = []
    batch_X_pos_act_name = []
    batch_X_pos_act_para_names = []

    bert_X_query = {'input_ids': [], 'input_mask': [], 'segment_ids': []}
    bert_X_pos_action_name = {'input_ids': [], 'input_mask': [], 'segment_ids': []}
    bert_X_pos_act_para_names = {'input_ids': [], 'input_mask': [], 'segment_ids': []}

    for q_vec, pos_action_name, pos_action_para_names, pos_action_para_dom, pos_act_bert_input in batch:

        batch_X_q.append(q_vec)

        batch_X_pos_act_name.append(pos_action_name)
        batch_X_pos_act_para_names.append(pos_action_para_names)

        # bert inputs
        add_bert_input(bert_X_query, pos_act_bert_input[0])
        add_bert_input(bert_X_pos_action_name, pos_act_bert_input[1])
        add_bert_input(bert_X_pos_act_para_names, pos_act_bert_input[2])

    action_batch = {'query': np.array(batch_X_q),
                    'pos_act_name': np.array(batch_X_pos_act_name),
                    'pos_act_para_names': np.array(batch_X_pos_act_para_names),

                    'bert_in_query': (np.array(bert_X_query['input_ids']),
                                      np.array(bert_X_query['input_mask']),
                                      np.array(bert_X_query['segment_ids'])),

                    'bert_in_pos_act_name': (np.array(bert_X_pos_action_name['input_ids']),
                                                np.array(bert_X_pos_action_name['input_mask']),
                                                np.array(bert_X_pos_action_name['segment_ids'])),
                    'bert_in_pos_act_para_names': (np.array(bert_X_pos_act_para_names['input_ids']),
                                                   np.array(bert_X_pos_act_para_names['input_mask']),
                                                   np.array(bert_X_pos_act_para_names['segment_ids'])),

                    }

    return action_batch


def get_next_test_batch_para_tagging(data, itr, batch_size):
    batch = data[(itr * batch_size):(itr * batch_size) + batch_size]

    # categorical_input
    batch_X_q = []
    batch_X_q_char = []
    batch_X_q_ent = []
    batch_X_tokens = []

    batch_X_pos_para_name = []
    batch_X_para_type = []

    batch_X_q_len = []

    bert_X_tagg_in = {'input_ids': [], 'input_mask': [], 'segment_ids': []}

    for q_vec, para_type, para_name, q_len, \
        q_char_vec, q_ent_vec, para_bert_input, bert_in_tokens, q_token_dict in batch:

        batch_X_q.append(q_vec)
        batch_X_q_char.append(q_char_vec)
        batch_X_q_ent.append(q_ent_vec)

        batch_X_pos_para_name.append(para_name)
        batch_X_para_type.append(para_type)

        batch_X_q_len.append(q_len)

        # bert inputs
        add_bert_input(bert_X_tagg_in, para_bert_input)
        batch_X_tokens.append((bert_in_tokens, q_token_dict))

    para_batch = {'query': np.array(batch_X_q), 'q_char': np.array(batch_X_q_char), 'q_ent': np.array(batch_X_q_ent),
                  'para_type': np.array(batch_X_para_type),
                  'pos_para_name': np.array(batch_X_pos_para_name),

                  'q_len': np.array(batch_X_q_len),

                  'bert_tagg_in': (np.array(bert_X_tagg_in['input_ids']),
                                   np.array(bert_X_tagg_in['input_mask']),
                                   np.array(bert_X_tagg_in['segment_ids'])),
                  }

    return para_batch, batch_X_tokens


def get_next_test_batch_para_matching(data, itr, batch_size):
    batch = data[(itr * batch_size):(itr * batch_size) + batch_size]

    batch_X_q_para_val = []
    batch_X_q_para_val_char = []

    batch_X_pos_para_name = []
    batch_X_para_type = []
    batch_X_pos_para_val = []
    batch_X_pos_para_val_char = []
    batch_X_pos_ext_match_score = []

    bert_X_pos_para_name = {'input_ids': [], 'input_mask': [], 'segment_ids': []}
    bert_X_pos_para_val = {'input_ids': [], 'input_mask': [], 'segment_ids': []}

    for q_para_val_vec, para_type, para_name, para_val, para_val_ext_match_score, \
        para_bert_input, q_para_val_char_vec, para_val_vec_char in batch:

        batch_X_q_para_val.append(q_para_val_vec)
        batch_X_q_para_val_char.append(q_para_val_char_vec)

        batch_X_pos_para_name.append(para_name)
        batch_X_para_type.append(para_type)
        batch_X_pos_para_val.append(para_val)
        batch_X_pos_para_val_char.append(para_val_vec_char)

        batch_X_pos_ext_match_score.append(para_val_ext_match_score)

        # bert inputs
        add_bert_input(bert_X_pos_para_name, para_bert_input[0])
        add_bert_input(bert_X_pos_para_val, para_bert_input[1])

    para_batch = {'q_para_val': np.array(batch_X_q_para_val),
                  'q_para_val_char': np.array(batch_X_q_para_val_char),

                  'para_type': np.array(batch_X_para_type),
                  'pos_para_name': np.array(batch_X_pos_para_name),

                  'pos_para_val': np.array(batch_X_pos_para_val),
                  'pos_para_val_char': np.array(batch_X_pos_para_val_char),
                  'pos_ext_match_score': np.array(batch_X_pos_ext_match_score),

                  'bert_in_pos_para_name': (np.array(bert_X_pos_para_name['input_ids']),
                                            np.array(bert_X_pos_para_name['input_mask']),
                                            np.array(bert_X_pos_para_name['segment_ids'])),
                  'bert_in_pos_para_val': (np.array(bert_X_pos_para_val['input_ids']),
                                           np.array(bert_X_pos_para_val['input_mask']),
                                           np.array(bert_X_pos_para_val['segment_ids'])),
                  }

    return para_batch
