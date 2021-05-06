from code.dataset_preparation.data_prep_util import *
from code.train_data_preprocessing.preprocess_util import get_candidate_query_phrases
np.random.seed(1234)


def get_phrase_from_tokens(pred_mention_tokens, q_token_dict):
    phrase = []
    i = 0
    visited_tokens = []
    for wd, wd_tokens in q_token_dict:
        if pred_mention_tokens[i:i+len(wd_tokens)] == wd_tokens:
            phrase.append(wd)
            i += len(wd_tokens)
            visited_tokens.extend(wd_tokens)
            if visited_tokens == pred_mention_tokens:
                break
        else:
           phrase = []
           visited_tokens = []

    return ' '.join(phrase)


def get_para_mentions_from_query(para_name_list, test_X_tokens, pred_st_index, pred_end_index):
    q_phrase_tag_dict = {}

    for i in range(len(test_X_tokens)):
        q_para_tokens = test_X_tokens[i][0]
        q_token_dict = test_X_tokens[i][1]

        para_name = para_name_list[i][0]
        para_type = para_name_list[i][1]

        st_index = pred_st_index[i]
        end_index = pred_end_index[i]

        pred_mention_tokens = q_para_tokens[st_index: (end_index+1)]

        ' =============================== '
        print('query:', q_para_tokens)
        print('para_name:', para_name)
        print('mention:', pred_mention_tokens)
        ' =============================== '

        if st_index == 0 and end_index == 0:
            extracted_phrase = ''
        else:
            extracted_phrase = get_phrase_from_tokens(pred_mention_tokens, q_token_dict)

        q_phrase_tag_dict[para_name] = (para_type, extracted_phrase)

    return q_phrase_tag_dict


def get_para_name_q_phrases(q_phrase, para_name_list, pred_tag_seq, pred_tag_score):
    q_phrase_tag_dict = {}
    q_words = q_phrase.split()

    for i in range(len(para_name_list)):
        para_name = para_name_list[i][0]
        para_type = para_name_list[i][1]

        tag_seq = pred_tag_seq[i]
        tag_score = pred_tag_score[i]

        print('query:', q_phrase)
        print('para_name:', para_name)
        print('tag_seq:', tag_seq[:len(q_words)])

        extracted_phrase = ' '.join([q_words[j] for j in range(len(q_words)) if tag_seq[j] == 1])

        q_phrase_tag_dict[para_name] = (para_type, extracted_phrase, tag_score)

    return q_phrase_tag_dict


def get_root_activity(node_DB):
    for activity_id in node_DB['activity']:
        if node_DB['activity'][activity_id]['IsHome'] == 'True':
            return activity_id, node_DB['activity'][activity_id]['ActivityName']
    return '-', '-'


def get_available_actions(activity_name, node_DB, para_DB, te_p_DB):
    action_desc_set = set()
    action_next_ids = set()
    for action_desc in node_DB['action']:
        if action_desc.startswith(activity_name+'->'):
            action_desc_set.add(action_desc)

    if len(action_desc_set) > 0:
        for action_desc in action_desc_set:
              for action_id in node_DB['action'][action_desc]:
                  if action_desc +'#'+ get_action_name(para_DB, action_id).replace('\'', ' ') in te_p_DB:
                      action_next_ids.add(action_id)
    return list(action_next_ids)


def get_available_action_parameters(q_phrase, selected_action_id, node_DB, para_DB, para_dom, ext_para_vals, te_p_DB):
    action_parameters = set()
    action_name = ()

    for para in para_DB[selected_action_id]:
        if para['Type'] == 'ACTION':
            action_name = (preprocess_text(para['description']), para['para_id'])
            if para['para_id'] in para_dom:
                 action_parameters.add((preprocess_text(para['description']), para['para_id'], 1))
        if para['Type'] == 'INPUT':
            action_parameters.add((preprocess_text(para['description']), para['para_id'], 1))
        if para['Type'] == 'TEXT':
            action_parameters.add((preprocess_text(para['description']), para['para_id'], 0))

    para_list = {}
    if len(action_parameters) == 0:
        if action_name[1] in para_dom:
            print("in para dom..")
            for phrase in para_dom[action_name[1]]:
                if action_name[0] in para_list:
                    para_list[action_name[0]].add((phrase.lower(), 1))
                else:
                    para_list[action_name[0]] = {(phrase.lower(), 1)}
    else:
        for para_name, para_id, para_type in action_parameters:

            if para_name.strip() in {'reservation date', 'date', 'check out', 'check in',
                                 'check out date', 'check in date',
                                 'check-out', 'check-in'}:
                para_list[para_name] = ext_para_vals['date']
            else:
                if para_id in para_dom and para_type == 1:
                    for phrase in para_dom[para_id]:
                        if para_name in para_list:
                            para_list[para_name].add((phrase.lower(), para_type))
                        else:
                            para_list[para_name] = {(phrase.lower(), para_type)}

                if para_type == 0:
                    q_uni_bigram_phrases = get_candidate_query_phrases(q_phrase)

                    for wd in q_uni_bigram_phrases:
                        if para_name in para_list:
                            para_list[para_name].add((wd, para_type))
                        else:
                            para_list[para_name] = {(wd, para_type)}

    return para_list


def get_action_desc(node_DB, action_id):
    for action_desc in node_DB['action']:
        if action_id in node_DB['action'][action_desc]:
            return action_desc
    return '-'


def get_action_name(para_DB, action_id):
    for para in para_DB[action_id]:
        if para['Type'] == 'ACTION':
            return para['description']
    return '-'


def get_activity_id(node_DB, activity_name):
    for activity_id in node_DB['activity']:
        if node_DB['activity'][activity_id]['ActivityName'] == activity_name:
            return activity_id
    return '-'


def extract_parameters(para_str):
    pred_para = {}

    if para_str != '':
        para_list = para_str.split(',')

        for para_tup in para_list:
            if '=' not in para_tup:
                print(para_str)
                input()
            pred_para[para_tup.split('=')[0].replace("\'", '').replace("\"", '').strip()] \
                = para_tup.split('=')[1].replace("\'", '').replace("\"", '').strip()
    return pred_para


def evaluate_performance(result_DB, result_file):
    act_res = []
    para_res = []
    para_res_prec = []
    para_res_rec = []
    para_res_prec_exact = []

    reject_count = 0

    for res_tup in result_DB:

        pred_path = res_tup[0]
        st_node = res_tup[1]
        gt_list = res_tup[2]
        reject = res_tup[3]

        if reject == 1:                # action not predicted
            reject_count += 1

            act_res.append(0)
            para_res_prec.append(0)
            para_res_rec.append(0)

            para_res.append(0)
            para_res_prec_exact.append(0)
        else:                                 # action predicted
            pred_action_name = preprocess_text(pred_path.split("{")[0].split(',')[1].replace('\'', ' ').strip())
            pred_para_dict = {}
            if len(pred_path.split("{")) > 1:
                pred_para_dict = extract_parameters(pred_path.split("{")[1].replace('}', ''))

            gold_action_para_dict = get_gold_act_para_list(st_node, gt_list)

            if pred_action_name in gold_action_para_dict:
               act_res.append(1)

               max_len = -1
               max_gold_para_tup_set = None
               max_pred_para_tup_set = None
               max_overlap_set = None

               gold_para_dict_list = gold_action_para_dict[pred_action_name]

               for gold_para_dict in gold_para_dict_list:
                   gold_para_tup_set = {(preprocess_text(para_name), preprocess_text(para_val))
                                        for para_name, para_val in gold_para_dict.items()}
                   pred_para_tup_set = {(preprocess_text(para_name), preprocess_text(para_val))
                                        for para_name, para_val in pred_para_dict.items()}

                   overlap_set = gold_para_tup_set.intersection(pred_para_tup_set)

                   if len(overlap_set) > max_len:
                       max_overlap_set = overlap_set
                       max_gold_para_tup_set = gold_para_tup_set
                       max_pred_para_tup_set = pred_para_tup_set
                       max_len = len(overlap_set)

               prec = 0.0
               rec = 0.0

               if len(max_pred_para_tup_set) > 0:
                    prec = (len(max_overlap_set) * 1.0) / len(max_pred_para_tup_set)

               if len(max_gold_para_tup_set) > 0:
                    rec = (len(max_overlap_set) * 1.0) / len(max_gold_para_tup_set)

               para_res_prec.append(prec)
               para_res_rec.append(rec)

               # EMA
               if prec == 1.0 and rec == 1.0:
                   para_res.append(1)
               else:
                   para_res.append(0)

               # PA-100
               if prec == 1.0:
                   para_res_prec_exact.append(1)
               else:
                   para_res_prec_exact.append(0)
            else:
               act_res.append(0)
               para_res_prec.append(0)
               para_res_rec.append(0)

               para_res.append(0)
               para_res_prec_exact.append(0)

    result_file.write("\n ======== \n")
    if len(act_res) > 0:
        print("Overall act accuracy: ", np.mean(act_res))
        result_file.write("Overall act accuracy: "+str(np.mean(act_res)))
    else:
        print("Overall act accuracy: 0.0")
        result_file.write("Overall act accuracy: 0.0 ")
    result_file.write("\n")

    if len(para_res_prec) > 0:
        para_prec_final = np.mean(para_res_prec)
        print("Overall parameter Prec: ", para_prec_final)
        result_file.write("Overall parameter Prec: "+str(para_prec_final))
    else:
        para_prec_final = 0.0
        print("Overall parameter Prec: 0.0")
        result_file.write("Overall parameter Prec: 0.0")
    result_file.write("\n")

    if len(para_res_prec) > 0:
        para_rec_final = np.mean(para_res_rec)
        print("Overall parameter rec: ", para_rec_final)
        result_file.write("Overall parameter rec: "+ str(para_rec_final))
    else:
        para_rec_final = 0.0
        print("Overall parameter rec: 0.0")
        result_file.write("Overall parameter rec: 0.0")
    result_file.write("\n")

    if para_prec_final > 0.0 and para_rec_final > 0.0:
        para_f1 = (2.0 * para_prec_final * para_rec_final) / (para_prec_final + para_rec_final)
        print("Overall parameter F1: ", para_f1)
        result_file.write("Overall parameter F1: "+str(para_f1))
    else:
        print("Overall parameter F1: 0.0")
        result_file.write("Overall parameter F1: 0.0")
    result_file.write("\n")

    if len(para_res) > 0:
        print("parameter exact match acc: ", np.mean(para_res))
        result_file.write("parameter EMA: "+str(np.mean(para_res)))
    else:
        print("parameter exact match acc: ", 0.0)
        result_file.write("parameter EMA: 0.0")
    result_file.write("\n")

    if len(para_res_prec_exact) > 0:
        print("parameter exact Prec acc: ", np.mean(para_res_prec_exact))
        result_file.write("PA-100: "+str(np.mean(para_res_prec_exact)))
    else:
        print("parameter exact Prec acc: ", 0.0)
        result_file.write("PA-100: 0.0")
    result_file.write("\n")

    print("reject count: ", reject_count)
    print("Total query: ", len(result_DB))


def get_gold_act_para_list(st_node, gt_list):
    act_para_dict = {}

    for gt in gt_list:
        if st_node == gt[0]:
            gold_act_name = preprocess_text(gt[1].replace('\'', ' ').strip())
            if gold_act_name in act_para_dict:
                act_para_dict[gold_act_name].append(gt[2])
            else:
                act_para_dict[gold_act_name] = [gt[2]]
    return act_para_dict
