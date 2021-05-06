import numpy as np, os
import pickle
from code.train_data_preprocessing.vectorize_dataset import get_vectorized_dataset
from code.dataset_preparation.data_prep_util import replace_punctuations
from code.train_data_preprocessing.preprocess_util import lemmatizer, get_candidate_query_phrases


def get_all_available_actions(activity_name, node_DB, para_DB):
    action_desc_set = set()
    action_names = set()
    for action_desc in node_DB['action']:
        if action_desc.startswith(activity_name+'->'):
            action_desc_set.add(action_desc)

    if len(action_desc_set) > 0:
        for action_desc in action_desc_set:
              for action_id in node_DB['action'][action_desc]:
                  for para in para_DB[action_id]:
                      if para['Type'] == 'ACTION':
                          action_names.add(action_desc +'#'+ para['description'])
    return action_names


def add_to_vocab(phrase, vocab_to_id, id_to_vocab):
    for wd in replace_punctuations(phrase).lower().split():
        wd = lemmatizer.lemmatize(wd)
        if wd not in vocab_to_id:
            vocab_to_id[wd] = len(vocab_to_id)
            id_to_vocab[vocab_to_id[wd]] = wd


def generate_pos_neg_examples(data, vocab_to_id, id_to_vocab, node_DB, para_DB, p_DB, update_vocab):

    if update_vocab:
        # add wild card ....
        add_to_vocab('null_action', vocab_to_id, id_to_vocab)
        add_to_vocab('null_para', vocab_to_id, id_to_vocab)
        add_to_vocab('null_val', vocab_to_id, id_to_vocab)
        add_to_vocab('stop', vocab_to_id, id_to_vocab)

    data_query_DB = []
    for q_inst, path_inst_set in data.items():

        act_path_seq = []
        para_path_seq = []

        all_pos_actions = set()
        for path_seq_inst in path_inst_set:
            for action, _ in path_seq_inst:
                all_pos_actions.add(action)

        un_inst_para_name_set = set()
        # positive training set ...
        for path_seq_inst in path_inst_set:
            for pos_action, pos_para_dict in path_seq_inst:

                ''' For action intent learning ...'''
                pos_action_name = pos_action.split("#")[1].strip()

                # update vocab
                if update_vocab:
                    add_to_vocab(pos_action_name, vocab_to_id, id_to_vocab)

                inst_para_name_set = set(pos_para_dict.keys())
                if pos_action in p_DB:
                    pos_para_name_set = set(p_DB[pos_action].keys())
                else:
                    pos_para_name_set = set()
                un_inst_para_name_set = un_inst_para_name_set.union(pos_para_name_set.difference(inst_para_name_set))

                # update vocab
                if update_vocab:
                    for pos_para_name in pos_para_name_set:
                        add_to_vocab(pos_para_name, vocab_to_id, id_to_vocab)

                curr_node = pos_action.split("->")[0].strip()

                # get all negative actions ...
                neg_act_list1 = []
                # other available actions in curr_node ...
                all_actions = get_all_available_actions(curr_node, node_DB, para_DB)

                neg_action_set = all_actions.difference(all_pos_actions)
                if len(neg_action_set) == 0:
                    neg_action_set.add(curr_node + '#null_action')

                for neg_action in neg_action_set:
                    neg_action_name = neg_action.split("#")[1].strip()
                    # update vocab
                    if update_vocab:
                        add_to_vocab(neg_action_name, vocab_to_id, id_to_vocab)

                    neg_para_name_set = []
                    if neg_action in p_DB:
                        for neg_para_name, _ in p_DB[neg_action].items():
                            neg_para_name_set.append(neg_para_name)

                    if len(neg_para_name_set) == 0:
                        neg_para_name_set.append('null_para')

                    # update vocab
                    if update_vocab:
                        for neg_para_name in neg_para_name_set:
                            add_to_vocab(neg_para_name, vocab_to_id, id_to_vocab)

                    neg_act_list1.append((neg_action, neg_para_name_set))

                act_path_seq.append((curr_node, pos_action, pos_para_name_set, neg_act_list1))

                '''  For parameter value learning ...  '''
                for pos_para_name, pos_para_tup in pos_para_dict.items():
                    pos_para_val = pos_para_tup[0]
                    pos_para_type = pos_para_tup[1]
                    pos_para_sample_val = pos_para_tup[2]

                    if pos_para_type == 0:
                        all_fixed_val = get_candidate_query_phrases(q_inst)   # get candidate noun phrases
                    else:
                        all_fixed_val = set([fixed_val for fixed_val, _, para_type in p_DB[pos_action][pos_para_name]])

                    neg_para_val_set = all_fixed_val.difference({pos_para_val})

                    if update_vocab:
                        add_to_vocab(pos_para_name, vocab_to_id, id_to_vocab)
                        add_to_vocab(pos_para_val, vocab_to_id, id_to_vocab)
                        for neg_val in neg_para_val_set:
                            add_to_vocab(neg_val, vocab_to_id, id_to_vocab)

                    if len(neg_para_val_set) == 0:
                        neg_para_val_set = {'null_val'}

                    para_path_seq.append((pos_action, pos_para_name, pos_para_val, pos_para_type,
                                          pos_para_sample_val, neg_para_val_set))

        add_to_vocab(q_inst.strip(), vocab_to_id, id_to_vocab)
        data_query_DB.append((q_inst, act_path_seq, para_path_seq, un_inst_para_name_set))

    return data_query_DB


def prepare_vectorized_dataset(trace_id, dataset_dump):

    vocab_to_id = {}
    id_to_vocab = {}
    train_data, valid_data, test_data, node_DB, para_DB, p_DB = dataset_dump

    # expand training dataset with pos and neg examples ...
    train_data_pos_neg = generate_pos_neg_examples(train_data, vocab_to_id, id_to_vocab, node_DB, para_DB, p_DB, update_vocab=True)
    print('pos neg example generation done for training ....')
    assert len(train_data_pos_neg) == len(train_data)
    valid_data_pos_neg = generate_pos_neg_examples(valid_data, vocab_to_id, id_to_vocab, node_DB, para_DB, p_DB, update_vocab=False)
    print('pos neg example generation done for valid ....')

    # generate dataset ..
    train = get_vectorized_dataset(train_data_pos_neg, vocab_to_id, p_DB)
    print('training data vectorized ....')
    valid = get_vectorized_dataset(valid_data_pos_neg, vocab_to_id, p_DB)
    print('valid data vectorized ....')

    print(np.array(train[0]).shape, '----\n')
    print(np.array(train[1]).shape, '----\n')
    print(np.array(train[2]).shape, '----\n')

    print('action training data shapes ----\n')
    for i in range(7):
        print(np.array(train[0][0][i]).shape, '----\n')

    print('para training data ..tag shapes  ----\n')
    for i in range(9):
        print(np.array(train[1][0][i]).shape, '----\n')

    print('para training data ...match shapes  ----\n')
    for i in range(10):
        print(np.array(train[2][0][i]).shape, '----\n')

    print('valid data stats ----\n')
    print(np.array(valid[0]).shape, '----\n')
    print(np.array(valid[1]).shape, '----\n')
    print(np.array(valid[2]).shape, '----\n')

    data_dump = (train, valid, vocab_to_id, id_to_vocab)

    print(os.getcwd())
    with open('../resource/'+ str(trace_id) + '_data_vec_dump.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data_dump, f)