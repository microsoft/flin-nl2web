import os

from code.navigation_module.test_query_read import *
from code.navigation_module.navigate_test_domain import navigate_domain
from code.dataset_preparation.website_para_reading import read_annotated_para_file
from code.dataset_preparation.load_additional_resources import load_external_resources
from code.navigation_module.util_navigation import evaluate_performance, get_root_activity


def run_evaluation(args, nsm_model, data_vec_dump):
    # loading training dataset and vocab ....
    _, _, vocab_to_id, id_to_vocab = data_vec_dump

    test_trace_id = args['test_trace_id']
    model_result_dict = args['model_result_dict']

    if not os.path.isdir('./all_results/' + model_result_dict + '/'):
        os.makedirs('./all_results/' + model_result_dict + '/')

    ''' call a test website schema parsing module here'''
    # node_DB, para_DB, para_dom, _, _, _ = get_processed_graph_info(test_trace_id)
    node_DB = None
    para_DB = None
    para_dom = None

    if args['eval_mode'] == 'of':

        basepath = args['data_path']

        # ext_data_path = None
        # result_file = None
        # if args['ext_test_set'] == 'd-flow':
        #     if args['ext_test_dom'] == 'S':
        #         ext_data_path = basepath + 'dialogflow_rei_pred_gt.csv'
        #         result_file = open(basepath+ 'result_d-flow_rei' + '.txt', 'w')
        #     elif args['ext_test_dom'] == 'R':
        #         ext_data_path = basepath + 'dialogflow_opentable_pred_gt.csv'
        #         result_file = open(basepath + 'result_d-flow_opentable' + '.txt', 'w')
        # else:
        #     if args['ext_test_dom'] == 'H':
        #         ext_data_path = basepath + 'google_hotel_pred_gt.csv'
        #         result_file = open(basepath+ 'result_google_hotel' + '.txt', 'w')
        # test_query_DB = read_gold_file(ext_data_path, q_type=args['test_query_type'], ext=True)

        # read from a file
        test_query_DB = read_gold_file(basepath+str(test_trace_id)+'_datafiles/'+
                                       str(test_trace_id)+'_test_q.csv', q_type=args['test_query_type'])
        result_basepath = './all_results/' + model_result_dict + '/'
        result_file = open(result_basepath + 'result_' + str(test_trace_id) + '.txt', 'w')

        q_id_cnt = 0

        # read from a file
        result_DB = []

        ext_KB = load_external_resources()
        ext_para_vals = {'date': []}
        for para_val_ext, paraphrase_set in ext_KB['date'].items():
            ext_para_vals['date'].append((para_val_ext, 1))


        te_p_DB = read_annotated_para_file(test_trace_id, args, ext_KB)
        print('# test queries: ', len(test_query_DB))
        total_queries = len(test_query_DB)

        for q_inst, gt_list in test_query_DB.items():

            st_activity_name = gt_list[0][0]

            navigation_path, navigation_path_id, \
            navigation_path_para_scores = navigate_domain(nsm_model, q_inst, node_DB, para_DB, para_dom,
                                                            vocab_to_id, st_activity_name,
                                                           ext_para_vals, te_p_DB, verbose=bool(args['verbose']))

            if len(navigation_path[1]) == 2:
                reject = 0
            else:
                reject = 1

            q_id_cnt += 1

            navigation_path_str = ','.join([str(element) for element in navigation_path[1]])
            print('Q-'+str(q_id_cnt)+' of '+ str(total_queries) +' : '+q_inst)
            print(navigation_path_str)
            result_file.write('Q-'+str(q_id_cnt)+': '+ q_inst+ '\n')
            result_file.write('gold path: '+ st_activity_name + ', ' + str(gt_list) + '\n')
            result_file.write('Pred PATH: ' + navigation_path_str + '\n')

            result_DB.append((navigation_path_str.strip(), st_activity_name, gt_list, reject))

        evaluate_performance(result_DB, result_file)
        result_file.close()

    elif args['eval_mode'] == 'i':
        # prepare test data ...
        trace_id = input('\n ======================== \nEnter TraceID: ')

        while not trace_id.isdigit():
            trace_id = input("Please Enter correct TraceID: ")

        print("OK....WAIT.....READING DOMAIN KNOwLEDGE!!!\n")
        result_DB = {}

        print("\n ****** AGENT is READY TO GO ..........\n")

        query_id = 1
        q_answerd = 0
        q_rejected = 0
        activity_name = ''

        while True:
            print("Enter 'exit' to stop.")
            q_phrase = input("User (Enter Query): ")

            if q_phrase == 'exit':
                break

            st_activity = input("User (Enter Start Activity): ")

            # get root activity ...
            if activity_name == '' and st_activity == '':
                _, activity_name = get_root_activity(node_DB)
            if st_activity != '':
                activity_name = st_activity

            navigation_path, navigation_path_id, \
            navigation_path_para_scores = navigate_domain(nsm_model, q_phrase, node_DB, para_DB, para_dom,
                                                            vocab_to_id, activity_name, verbose=bool(args['verbose']))
            # input()
            if len(navigation_path) > 0 and len(navigation_path[1]) > 1:
                activity_name = navigation_path[1][-1].split("#")[0].split("->")[1].strip()

            if len(navigation_path) > 0 and len(navigation_path[1]) > 0:
                print('PATH:', navigation_path[1])
                result_DB[query_id] = (navigation_path, navigation_path_id)
                navigation_path_str = ' ; '.join([str(element) for element in navigation_path[1]])
                print(navigation_path_str)

                q_answerd += 1
            else:
                q_rejected += 1

            query_id += 1

