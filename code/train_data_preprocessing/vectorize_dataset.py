from code.dataset_preparation.data_prep_util import *
from code.train_data_preprocessing.bert_preprocess_util import *
from code.train_data_preprocessing.preprocess_util import get_vectorized_phrase, pad_arr_seq,\
                                 get_vectorized_char_seq, char_vocab_to_id, ent_vocab_to_id, \
                                 get_vectorized_entity_tags, get_gold_labels_tagger
from code.globals import max_no_dict, max_seq_len
from code.train_data_preprocessing.external_matching_module import get_ext_matching_score


def get_vectorized_dataset(data_pos_neg, vocab_to_id, p_DB):
    '''
                        Input: batch_size x [q=[15], node_desc=[max_no_phrase x 20], node_content=[max_#_content x 10],
                                          act_name= [5], act_para_phrase: [max_#_parameter x 5]
                                          act_para_dom: [max_#_parameter x dom_size x 7]]
                        :return:
    '''

    act_data = []
    para_data_tag = []
    para_data_match = []

    count = 0
    total_queries = len(data_pos_neg)

    for q_phrase, act_path_seq, para_path_seq, un_inst_para_name_set in data_pos_neg:

        for action_tr_tup in act_path_seq:
            act_data_tuple_vec = get_act_data_tuple_vec(q_phrase, action_tr_tup, vocab_to_id, p_DB)
            act_data.append(act_data_tuple_vec)

        for para_tr_tup in para_path_seq:
            para_data_tuple_vec_1 = get_para_data_tuple_vec_tagging(q_phrase, para_tr_tup, vocab_to_id)
            para_data_tuple_vec_2 = get_para_data_tuple_vec_matching(para_tr_tup, vocab_to_id)

            para_data_tag.append(para_data_tuple_vec_1)
            para_data_match.append(para_data_tuple_vec_2)

        for un_inst_para_name in un_inst_para_name_set:
            para_tr_tup = ('-', un_inst_para_name, 'EMPTY', 1, 'NULL', {})
            para_data_tuple_vec_1 = get_para_data_tuple_vec_tagging(q_phrase, para_tr_tup, vocab_to_id)
            para_data_tag.append(para_data_tuple_vec_1)

        count += 1

        if count % 400 == 0:
            print ('{} of {} queries have been vectorized'.format(count, total_queries))

    return (act_data, para_data_tag, para_data_match)


def get_action_vectors(action_name, action, para_name_set, p_DB, vocab_to_id):
    # action name & parameters [curr action]....
    action_name_vec = get_vectorized_phrase(action_name, vocab_to_id, max_seq_len['action_name'])  # 5
    action_para_names_vec = []  # 10 x 5    [10 parameters, each para name can be of length 5]
    action_para_dom_vec = []  # 10 x 15 x 7  [10 parameters, each para can take max. 15 values, each value's length= 7]

    para_type = 0
    for para_name in para_name_set:

        action_para_names_vec.append(
            get_vectorized_phrase(para_name, vocab_to_id, max_seq_len['para_name']))  # 5

        action_para_dom_values_vec = []  # 15 x 7
        if action in p_DB and para_name in p_DB[action]:
            dom_set = set([fixed_val for fixed_val, _, para_type in p_DB[action][para_name]])
            if len(dom_set) > 0:
                dom_sample = np.random.choice(list(dom_set), min(len(dom_set), max_no_dict['domain_size_per_para']),
                                              replace=False)
                for dom_val_phrase in dom_sample:
                    action_para_dom_values_vec.append(
                        get_vectorized_phrase(dom_val_phrase, vocab_to_id, max_seq_len['para_val']))

        action_para_dom_values_vec = pad_arr_seq(action_para_dom_values_vec,
                                                   max_no_dict['domain_size_per_para'],
                                                   [0] * max_seq_len['para_val'])
        action_para_dom_vec.append(action_para_dom_values_vec)

    # padding ...
    action_para_names_vec = pad_arr_seq(action_para_names_vec, max_no_dict['max_no_para_per_action'],
                                          [0] * max_seq_len['para_name'])

    action_para_dom_vec = pad_arr_seq(action_para_dom_vec, max_no_dict['max_no_para_per_action'],
                                       [[0] * max_seq_len['para_val']] * max_no_dict[
                                           'domain_size_per_para'])

    return action_name_vec, action_para_names_vec, \
           action_para_dom_vec


def get_act_data_tuple_vec(q_phrase, action_tr_tup, vocab_to_id, p_DB):
    # extract info from action training tuple
    curr_node, pos_action, pos_para_name_set, neg_act_list1 = action_tr_tup
    pos_action_name = pos_action.split("#")[1].strip()

    q_vec = get_vectorized_phrase(q_phrase, vocab_to_id, max_seq_len['query'])

    pos_action_name_vec, pos_action_para_names_vec, \
    pos_action_para_dom_vec = get_action_vectors(pos_action_name, pos_action,
                                                      pos_para_name_set, p_DB, vocab_to_id)

    ''' === pos bert input p1 ===== '''
    bert_in_query = get_vectorized_bert_input_phrase(q_phrase, max_seq_len['bert_query'])
    bert_in_pos_action_name = get_vectorized_bert_input_phrase(pos_action_name, max_seq_len['bert_action_name'])

    pos_act_para_names_str = ';'.join([para_name for para_name in pos_para_name_set])
    bert_in_pos_action_para_names = get_vectorized_bert_input_phrase(pos_act_para_names_str, max_seq_len['bert_para_names_str'])

    bert_in_pos = (bert_in_query, bert_in_pos_action_name, bert_in_pos_action_para_names)
    ''' =========================== '''

    neg_act_list_vec = []
    bert_in_neg_list = []

    # phase 1 neg examples ....
    for neg_action, neg_para_name_set in neg_act_list1:
        neg_action_name = neg_action.split("#")[1].strip()

        neg_action_name_vec, neg_action_para_names_vec, \
        neg_action_para_dom_vec = get_action_vectors(neg_action_name, neg_action,
                                                     neg_para_name_set, p_DB, vocab_to_id)

        neg_act_list_vec.append((neg_action_name_vec, neg_action_para_names_vec, neg_action_para_dom_vec))

        ''' === neg bert input p1 ====='''
        bert_in_neg_action_name = get_vectorized_bert_input_phrase(neg_action_name, max_seq_len['bert_action_name'])

        neg_act_para_names_str = ';'.join([neg_para_name for neg_para_name in neg_para_name_set])
        bert_in_neg_action_para_names = get_vectorized_bert_input_phrase(neg_act_para_names_str,
                                                                         max_seq_len['bert_para_names_str'])

        bert_in_neg = (bert_in_neg_action_name, bert_in_neg_action_para_names)
        bert_in_neg_list.append(bert_in_neg)
        ''' ==========================='''

    data_vec_tup = (q_vec, pos_action_name_vec, pos_action_para_names_vec, pos_action_para_dom_vec,
                     neg_act_list_vec, bert_in_pos, bert_in_neg_list)

    return data_vec_tup


def get_para_data_tuple_vec_tagging(q_phrase, para_tr_tup, vocab_to_id):
    para_tup_list = []

    # extract info from para_tr_tup
    _, pos_para_name, _, pos_para_type, pos_para_val_sample, neg_para_val_set = para_tr_tup

    # pos para val representations ...
    pos_para_name_vec = get_vectorized_phrase(pos_para_name, vocab_to_id, max_seq_len['para_name'])  # 5

    q_vec = get_vectorized_phrase(q_phrase, vocab_to_id, max_seq_len['query'])  # 15
    q_char_vec = get_vectorized_char_seq(q_phrase, char_vocab_to_id, max_seq_len['query'], max_seq_len['word']) # 15 x 10
    q_ent_vec = get_vectorized_entity_tags(q_phrase, ent_vocab_to_id, max_seq_len['query']) # 15

    label_vec, q_len = get_gold_labels_tagger(q_phrase, pos_para_val_sample, max_seq_len['query'])

    bert_in_tagging, gold_label_ids = get_bert_input_query_para_name(q_phrase, pos_para_name,
                                                      pos_para_val_sample, max_seq_len['bert_query_para_name'])

    para_tup = (q_vec, pos_para_type, pos_para_name_vec,
                label_vec, q_len, q_char_vec, q_ent_vec, bert_in_tagging, gold_label_ids)

    return para_tup


def get_para_data_tuple_vec_matching(para_tr_tup, vocab_to_id):

    # extract info from para_tr_tup
    _, pos_para_name, pos_para_val, pos_para_type, pos_para_val_sample, neg_para_val_set = para_tr_tup

    q_para_val_vec = get_vectorized_phrase(pos_para_val_sample, vocab_to_id, max_seq_len['para_val'])  # 15
    q_para_val_vec_char = get_vectorized_char_seq(pos_para_val_sample, char_vocab_to_id, max_seq_len['para_val'],
                                                    max_seq_len['word'])  # 7 x 10

    # pos para val representations ...
    pos_para_name_vec = get_vectorized_phrase(pos_para_name, vocab_to_id, max_seq_len['para_name'])  # 5
    pos_para_val_vec = get_vectorized_phrase(pos_para_val, vocab_to_id, max_seq_len['para_val'])  # 7
    pos_para_val_vec_char = get_vectorized_char_seq(pos_para_val, char_vocab_to_id, max_seq_len['para_val'], max_seq_len['word'])  # 7 x 10

    # lexical matching score for pos value
    pos_val_ext_match_score = get_ext_matching_score(pos_para_val_sample, pos_para_val)

    ''' === pos bert input p2 ====='''
    bert_in_pos_para_name = get_vectorized_bert_input_phrase(pos_para_name, max_seq_len['bert_para_name'])
    bert_in_pos_para_val = get_vectorized_bert_input_phrase(pos_para_val, max_seq_len['bert_para_val'])

    pos_para_bert_input = (bert_in_pos_para_name, bert_in_pos_para_val)

    ''' ======================== '''

    # para val training neg examples ....
    neg_para_val_list_vec = []
    neg_para_bert_input_list = []

    for neg_val in neg_para_val_set:
        neg_para_val_vec = get_vectorized_phrase(neg_val, vocab_to_id, max_seq_len['para_val'])  # 7
        neg_para_val_vec_char = get_vectorized_char_seq(neg_val, char_vocab_to_id, max_seq_len['para_val'], max_seq_len['word'])  # 7 x 10

        # lexical matching score for neg value
        neg_val_ext_match_score = get_ext_matching_score(pos_para_val_sample, neg_val)

        neg_para_val_list_vec.append((neg_para_val_vec, neg_val_ext_match_score, neg_para_val_vec_char))

        ''' === neg bert input p2 ====='''
        bert_in_neg_para_val = get_vectorized_bert_input_phrase(neg_val, max_seq_len['bert_para_val'])
        # neg_bert_input_p2 = get_vectorized_bert_input_p2(q_phrase, pos_para_name, neg_val)

        neg_para_bert_input_list.append(bert_in_neg_para_val)
        ''' ======================== '''

    para_tup = (q_para_val_vec, pos_para_type, pos_para_name_vec, pos_para_val_vec, pos_val_ext_match_score,
                neg_para_val_list_vec, pos_para_bert_input, neg_para_bert_input_list,
                q_para_val_vec_char, pos_para_val_vec_char)
    return para_tup