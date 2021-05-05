from code.navigation_module.util_navigation import *
from code.train_data_preprocessing.bert_preprocess_util import *
from code.train_data_preprocessing.preprocess_util import get_vectorized_phrase, get_vectorized_char_seq, \
                                            get_vectorized_entity_tags, char_vocab_to_id, ent_vocab_to_id
from code.train_data_preprocessing.external_matching_module import get_ext_matching_score
from code.train_data_preprocessing.preprocess_util import pad_arr_seq
from code.globals import max_seq_len, max_no_dict


def get_test_tuple_vec_act(q_vec, q_phrase, action_next, para_DB, para_dom, vocab_to_id):
    '''  test tup vector for action prediction  '''

    para_name_set = []
    action_name = ''

    ''' ================= action [next] vector generation =====================  '''
    # action name & parameters [curr action]....
    action_name_vec = pad_arr_seq([], max_seq_len['action_name'], 0)  # 5
    action_para_names_vec = []   # 10 x 5
    action_para_dom_vec = []      # 10 x 15 x 7

    if action_next in para_DB:
        for para in para_DB[action_next]:
            if para['Type'] == 'ACTION':
                action_name = preprocess_text(para['description'])
                action_name_vec = get_vectorized_phrase(preprocess_text(para['description']),
                                                        vocab_to_id, max_seq_len['action_name'])         # 5
            else:
                para_name_set.append(para['description'])
                action_para_names_vec.append(get_vectorized_phrase(para['description'], vocab_to_id, max_seq_len['para_name']))  # 5

                action_para_dom_val_vec = []     # 15 x 7
                if para['para_id'] in para_dom:
                    dom_set = para_dom[para['para_id']]

                    if len(dom_set) > 0:
                        dom_sample = np.random.choice(list(dom_set), min(len(dom_set), max_no_dict['domain_size_per_para']), replace=False)
                        for dom_val in dom_sample:
                            action_para_dom_val_vec.append(get_vectorized_phrase(dom_val, vocab_to_id, max_seq_len['para_val']))

                action_para_dom_val_vec = pad_arr_seq(action_para_dom_val_vec, max_no_dict['domain_size_per_para'],
                                                      [0] * max_seq_len['para_val'])
                action_para_dom_vec.append(action_para_dom_val_vec)

    action_para_names_vec = pad_arr_seq(action_para_names_vec, max_no_dict['max_no_para_per_action'],
                                          [0] * max_seq_len['para_name'])

    action_para_dom_vec = pad_arr_seq(action_para_dom_vec, max_no_dict['max_no_para_per_action'],
                                          [[0] * max_seq_len['para_val']] * max_no_dict['domain_size_per_para'])

    ''' === pos bert input p1 ===== '''
    bert_in_query = get_vectorized_bert_input_phrase(q_phrase, max_seq_len['bert_query'])
    bert_in_action_name = get_vectorized_bert_input_phrase(action_name, max_seq_len['bert_action_name'])

    act_para_names_bert = ';'.join([para_name for para_name in para_name_set])
    bert_in_action_para_names = get_vectorized_bert_input_phrase(act_para_names_bert, max_seq_len['bert_para_names_str'])

    act_bert_input = (bert_in_query, bert_in_action_name, bert_in_action_para_names)

    ''' =========================== '''

    data_vec_tup = (q_vec, action_name_vec, action_para_names_vec, action_para_dom_vec, act_bert_input)
    return data_vec_tup


def get_test_tuple_vec_para_tagging(q_vec, q_phrase, q_len, para_name, para_type, vocab_to_id):
    '''  test tup vector for parameter value prediction  '''

    # para val representations ...
    para_name_vec = get_vectorized_phrase(para_name, vocab_to_id, max_seq_len['para_name'])  # 5

    # para val specific query representations ....
    q_char_vec = get_vectorized_char_seq(q_phrase, char_vocab_to_id, max_seq_len['query'], max_seq_len['word'])  # 15 x 10
    q_ent_vec = get_vectorized_entity_tags(q_phrase, ent_vocab_to_id, max_seq_len['query'])  # 15

    para_bert_input, bert_in_tokens, q_token_dict = get_bert_input_query_para_name_test(q_phrase, para_name,
                                                                        max_seq_len['bert_query_para_name'])

    para_tup = (q_vec, para_type, para_name_vec, q_len,
                q_char_vec, q_ent_vec, para_bert_input, bert_in_tokens, q_token_dict)

    return para_tup


def get_test_tuple_vec_para_matching(q_para_phrase, para_name, para_type, para_val, vocab_to_id):
    '''  test tup vector for parameter value prediction  '''

    q_para_val_vec = get_vectorized_phrase(q_para_phrase, vocab_to_id, max_seq_len['para_val'])  # 15
    q_para_val_char_vec = get_vectorized_char_seq(q_para_phrase, char_vocab_to_id,
                                                  max_seq_len['para_val'],  max_seq_len['word'])  # 15 x 10

    # para val representations ...
    para_name_vec = get_vectorized_phrase(para_name, vocab_to_id, max_seq_len['para_name'])  # 5
    para_val_vec = get_vectorized_phrase(para_val, vocab_to_id, max_seq_len['para_val'])  # 7
    para_val_vec_char = get_vectorized_char_seq(para_val, char_vocab_to_id, max_seq_len['para_val'], max_seq_len['word'])  # 7 x 10

    # lexical matching score for pos value
    para_val_ext_match_score = get_ext_matching_score(q_para_phrase, para_val)

    ''' === pos bert input p2 ====='''
    bert_in_para_name = get_vectorized_bert_input_phrase(para_name, max_seq_len['bert_para_name'])
    bert_in_para_val = get_vectorized_bert_input_phrase(para_val, max_seq_len['bert_para_val'])

    para_bert_input = (bert_in_para_name, bert_in_para_val)
    ''' ======================== '''

    para_tup = (q_para_val_vec, para_type, para_name_vec, para_val_vec, para_val_ext_match_score,
                para_bert_input, q_para_val_char_vec,  para_val_vec_char)

    return para_tup