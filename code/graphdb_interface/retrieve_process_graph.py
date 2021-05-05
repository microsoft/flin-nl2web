from code.dataset_genaration.load_additional_resources import load_external_resources
from code.graphdb_interface.fetch_info_graph_db import *


def get_processed_graph_info(trace_id, external_resource_dir='./'):
    num_nodes = get_node_count(trace_id, "")
    print("Node Count: ", num_nodes)

    ext_KB = load_external_resources(external_resource_dir)

    # collect transition nodes and their properties ...
    node_DB, para_DB = get_all_nodes_properties(trace_id)

    for node_type in node_DB:
        print(node_type, len(node_DB[node_type]), node_DB[node_type])

    # get all contents ...
    content_DB = get_all_activity_content(trace_id, node_DB)
    para_dom = get_parameters_content(trace_id, ext_KB, para_DB)
    # print('para_dom: ', para_dom)

    for activity_id in node_DB['activity']:
        if activity_id not in content_DB:
            content_DB[activity_id] = {'page_content': {}, 'page_description':{}}

    idf_map = {}
    # filter with tf-idf ...
    for activity_id in content_DB:
        activity_name = node_DB['activity'][activity_id]['ActivityName']

        if 'page_content' in content_DB[activity_id]:
            for phrase in content_DB[activity_id]['page_content']:
                if phrase in idf_map:
                   idf_map[phrase].add(activity_name)
                else:
                   idf_map[phrase] = {activity_name}
        else:
            print(activity_id, 'page content empty **************')

    for phrase, activity_set in idf_map.items():
        if len(activity_set) > len(content_DB)/4:
            print(phrase, activity_set)

            for activity_id in content_DB:
                for text_tag in content_DB[activity_id]:
                    if text_tag != 'page_type':
                        if phrase in content_DB[activity_id][text_tag]:
                             content_DB[activity_id][text_tag].remove(phrase)

    # remove action content from activities ...
    for activity_id in content_DB:
        print("\n == activity_name: ", node_DB['activity'][activity_id]['ActivityName'])
        act_cont = get_action_content(para_dom, node_DB, para_DB, activity_id, 'out')
        print('out action_content : ', act_cont)

        for text_tag in content_DB[activity_id]:
            if text_tag != 'page_type':
                 for phrase in act_cont:
                     if phrase in content_DB[activity_id][text_tag]:
                         content_DB[activity_id][text_tag].remove(phrase)
                 print('filtered activity_content : ', content_DB[activity_id][text_tag], text_tag)
                 #input()

    # print content of an activity
    root_node = ''
    dest_nodes = []

    for activity_id in content_DB:
        print("\n activity_content: ", node_DB['activity'][activity_id]['ActivityName'])
        if node_DB['activity'][activity_id]['IsHome'] == 'True':
            root_node = node_DB['activity'][activity_id]['ActivityName']
        else:
            dest_nodes.append(node_DB['activity'][activity_id]['ActivityName'])

        for text_tag in content_DB[activity_id]:
            print('\n')
            print(text_tag, ':', content_DB[activity_id][text_tag])

    for action_desc in node_DB['action']:
         print("action_para: ")
         for action_id in node_DB['action'][action_desc]:
             print(action_desc, action_id)
             for para in para_DB[action_id]:
                 print("***", para)
                 if para['para_id'] in para_dom:
                    print(para, para_dom[para['para_id']])

    print(root_node)
    print(dest_nodes)

    return node_DB, para_DB, para_dom, content_DB, root_node, dest_nodes


if __name__ == '__main__':
    trace_id = '1086' # trace ids are listed in website_index.txt
    node_DB, para_DB, para_dom, content_DB, root_node, dest_nodes = get_processed_graph_info(trace_id)
