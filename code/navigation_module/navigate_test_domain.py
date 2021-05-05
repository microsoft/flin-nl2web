from operator import itemgetter

from code.globals import max_seq_len
from code.navigation_module.test_batch_util import get_next_test_batch_action, \
                                     get_next_test_batch_para_tagging, get_next_test_batch_para_matching
from code.navigation_module.util_navigation import *
from code.navigation_module.vectorize_test_data import get_test_tuple_vec_act, \
                                        get_test_tuple_vec_para_tagging, get_test_tuple_vec_para_matching
from code.train_data_preprocessing.preprocess_util import get_vectorized_phrase


def navigate_domain(nsm_model, q_phrase, node_DB, para_DB, para_dom, vocab_to_id, activity_name,
                    ext_para_vals, te_p_DB, verbose, use_additional_rules = False):

    q_vec = get_vectorized_phrase(q_phrase, vocab_to_id, max_seq_len['query'])
    q_len = len(q_phrase.split())
    activity_id = get_activity_id(node_DB, activity_name)

    explored_actions = set()
    navigation_path = {1: [activity_name], 2: [activity_name], 3: [activity_name]}
    navigation_path_id = [activity_id]
    navigation_path_para_scores = []

    while True:
        #  =============== IF NO STOP, PICK NEXT ACTION and FILL PARAMETERS ===========================

        action_next_list = get_available_actions(activity_name, node_DB, para_DB, te_p_DB)
        action_next_list = list(set(action_next_list).difference(explored_actions))

        if len(action_next_list) == 0:
            break

        # create test set ...
        test_data_act = []
        for action_next in action_next_list:
            test_tup = get_test_tuple_vec_act(q_vec, q_phrase, action_next, para_DB, para_dom, vocab_to_id)
            test_data_act.append(test_tup)

        test_X_action_batch = get_next_test_batch_action(test_data_act, 0, len(test_data_act))

        # action prediction ....
        act_pred_scores = nsm_model.predict_action(test_X_action_batch)

        # score actions ...
        action_scores = {}
        i = 0
        for action_id in action_next_list:
            action_desc = get_action_desc(node_DB, action_id)
            action_name = get_action_name(para_DB, action_id)

            action_scores[action_id] = (action_desc + '#' + action_name, 1.0 - act_pred_scores[i])
            i += 1

        if verbose:
            print("Query: ", q_phrase)
            print('pred_score:', act_pred_scores)
            print(action_scores)

        # ================= PARA VALUE PREDICTION ==========================

        action_para_pred = {}

        # looping through all actions
        for action_id in action_next_list:
            pred_para = {}
            para_list = get_available_action_parameters(q_phrase, action_id, node_DB, para_DB, para_dom, ext_para_vals, te_p_DB)

            if len(para_list) == 0:
                action_name_str = get_action_name(para_DB, action_id)
                pred_para[action_name_str] = (1, action_name_str, action_scores[action_id][1])
            else:
                # if verbose:
                #     print("para_list: ", para_list)
                # e.g., para_list: {'popular restaurants': {('heritage restaurant', 1), ('fuji steak house', 1),...}}

                # looping through every parameter of an action (e.g., first date, then time, then people, etc.)
                test_data_para_tag = []
                para_name_list = []
                for para_name, para_dom_vals in para_list.items():

                    para_val_list = list(para_dom_vals)
                    para_type = para_val_list[0][1]

                    # create test set ...
                    test_tup = get_test_tuple_vec_para_tagging(q_vec, q_phrase, q_len, para_name, para_type, vocab_to_id)
                    test_data_para_tag.append(test_tup)
                    para_name_list.append((para_name, para_type))

                test_X_para_batch_tag, test_X_tokens = get_next_test_batch_para_tagging(test_data_para_tag,
                                                                                        0, len(test_data_para_tag))

                # para tagging ....
                pred_st_index, pred_end_index = nsm_model.predict_para_tag(test_X_para_batch_tag)
                q_phrase_tag_dict = get_para_mentions_from_query(para_name_list, test_X_tokens, pred_st_index, pred_end_index)

                if verbose:
                    print("Parameter extraction: ", q_phrase_tag_dict)

                # ==========================================================

                for para_name, para_tag_tup in q_phrase_tag_dict.items():

                    para_val_list2 = list(para_list[para_name])

                    para_type = para_tag_tup[0]
                    q_para_phrase = para_tag_tup[1]

                    if q_para_phrase == '':
                        continue

                    if para_type == 0:
                        pred_para[para_name] = (para_type, q_para_phrase, 1.0)
                    else:

                        # create test set ...
                        test_data_para_match = []

                        for para_val, para_type in para_val_list2:
                            test_tup = get_test_tuple_vec_para_matching(q_para_phrase, para_name, para_type, para_val, vocab_to_id)
                            test_data_para_match.append(test_tup)

                        test_X_para_batch_matching = get_next_test_batch_para_matching(test_data_para_match, 0, len(test_data_para_match))

                        para_pred_scores_match = nsm_model.predict_para_match(test_X_para_batch_matching)

                        para_score_dict = {}
                        para_index = 0
                        for para_val, para_type in para_val_list2:
                            para_score_dict[para_val] = 1.0 - list(para_pred_scores_match)[para_index]
                            para_index += 1

                        if verbose:
                            print('Parameter scores for ', para_name)
                            print('Extracted phrase ', q_para_phrase)
                            print('para_score_dict ', sorted(para_score_dict.items(), key=itemgetter(1), reverse=True))
                            print('\n')

                        pred_para_val = max(para_score_dict.items(), key=itemgetter(1))[0]
                        max_pred_score = para_score_dict[pred_para_val]

                        pred_thresh = 0.67

                        if max_pred_score >= pred_thresh:
                            pred_para[para_name] = (para_type, pred_para_val, max_pred_score)

            if verbose:
                print('predicted parameters before filtering: ', pred_para)

            action_para_pred[action_id] = pred_para

        ''' ================= FINAL ACTION PREDICTION  =========== '''
        action_net_scores = []
        for action_id in action_next_list:

            para_score = []
            para_dict = {}
            for para_name, para_tup in action_para_pred[action_id].items():
                para_type = para_tup[0]
                para_dict[para_name] = para_tup[1]
                if para_type == 1:
                   para_score.append(para_tup[2])
                else:
                   para_score.append(1.0)

            if len(para_score) > 0:
                mean_para_score = np.mean(para_score)

                net_score = 0.4 * action_scores[action_id][1] + 0.6 * mean_para_score
                action_net_scores.append((action_scores[action_id][0], para_dict, net_score, action_id, mean_para_score,
                                          action_scores[action_id][1]))

        if len(action_net_scores) > 0:
            action_net_scores.sort(key=itemgetter(2), reverse=True)

            print('action_net_scores:', action_net_scores)

            for action_ranked in action_net_scores:
                print(action_ranked[0], action_ranked[1],   ' para_score: ', action_ranked[4],
                                                            ' act_score: ', action_ranked[5],  ' net_score: ', action_ranked[2])
            if action_net_scores[0][2] > 0.0:
                selected_action_desc = action_net_scores[0][0].split("#")[0]
                selected_action_name = action_net_scores[0][0].split("#")[1]
                selected_pred_para = action_net_scores[0][1]
                selected_action_id = action_net_scores[0][3]
                selected_pred_para_scores = action_para_pred[selected_action_id]
            else:
                selected_action_desc = activity_name+'->END'
                selected_action_name = 'STOP'
                selected_pred_para = {}
                selected_action_id = -1
                selected_pred_para_scores = {}

            if selected_action_name != 'STOP':
                navigation_path[1].append(selected_action_desc + "#" +
                                       selected_action_name + str(selected_pred_para).replace('\"', '\'').replace('\':', '\'='))
                navigation_path_para_scores.append(selected_pred_para_scores)
                navigation_path_id.append(selected_action_id)
                explored_actions.add(selected_action_id)

            if selected_action_name == 'STOP':
                break
            activity_name = selected_action_desc.split('->')[1]
            activity_id = get_activity_id(node_DB, activity_name)

        navigation_path_id.append(activity_id)
        if verbose:
            print("Query:", q_phrase)
            print(navigation_path)
            print(" ====================================== \n")

        if len(navigation_path) > 1:
            break

    return navigation_path, navigation_path_id, \
           navigation_path_para_scores

