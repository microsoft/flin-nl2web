from gremlin_python.driver import client, serializer
from code.graphdb_interface.graphdb_util import *
from code.dataset_genaration.data_gen_util import replace_punctuations

# initialize graph db ...
# add address to the graph database with Gremlin API where data is stored and credentials
client = client.Client('', 'g', username="", password="", message_serializer=serializer.GraphSONSerializersV2d0())

activityId = 'activityId'
aId = 'ActionId'
pId = 'parameterId'

# node property set
property_set = {'activity': {'ApplicationName', 'ActivityName', 'IsHome', 'snapshotCount'}}  #node_DB
para_prop = {'parameter': {'description', 'Type', aId}}    # para_DB

NO_text_set = {'text', 'any', 'visibile', 'html', 'immutables', 'entity', 'item-list-element', 'breadcrumb-list',
               'schemaorg', 'itemlistelement', 'breadcrumblist', 'position', 'item', 'name'}


def get_node_count(traceid, node_type):
    if node_type == "":
        query = 'g.V().has("TraceId",\"' + str(traceid) + '\").count()'
    else:
        query = 'g.V().has("TraceId",\"' + str(traceid) + '\").haslabel("'+node_type+'").count()'

    result_list = execute_query(query, client)
    if result_list is not None:
       return result_list.next()
    else:
       return 0


def get_all_nodes_properties(traceid):
    query = 'g.v().has("TraceId", "'+traceid+'").valueMap(true)'
    result_list = execute_query(query, client)

    node_DB = {'action':{}}             # node_type: {node_id: node_properties}
    para_DB = {}                        # action_id: [para1_dict, para2_dict, .....]
    for result in result_list:
        for prop in result:
            if prop['label'] in property_set:
                prop_dict = get_property(prop, property_set, prop['label'])
                if prop['label'] in node_DB:
                    node_DB[prop['label']][prop[activityId][0]] = prop_dict
                else:
                    if prop['label'] in property_set:
                        node_DB[prop['label']] = {prop[activityId][0]: prop_dict}

                if '4b7f4188-3f39-da1e-98ff-313ce993acd9' in node_DB['activity']:
                    node_DB['activity']['4b7f4188-3f39-da1e-98ff-313ce993acd9']['IsHome'] = 'False'

            # action ...
            if prop['label'] == 'action':
                if prop['description'][0] in node_DB['action']:
                    if traceid == '925':
                         node_DB['action'][prop['description'][0].replace('#', '')].add(prop['id'])
                    else:
                         node_DB['action'][prop['description'][0].replace('#', '')].add(prop[aId][0])
                else:
                    if traceid == '925':
                         node_DB['action'][prop['description'][0].replace('#', '')] = {prop['id']}
                    else:
                         node_DB['action'][prop['description'][0].replace('#', '')] = {prop[aId][0]}

            # 'parameter': action_id:{'description', 'Type', 'paraId'}
            if prop['label'] == 'parameter':
                prop_dict = get_property(prop, para_prop, 'parameter')

                if 'Type' in prop_dict and (prop_dict['Type'] == 'DEEPLINK' or prop_dict['Type'] == 'PATHKEY'):
                    continue

                text_str = prop_dict['description'].lower()
                if "|" in text_str:
                    text_str = text_str.split("|")[0]

                text_str2 = []
                for wd in text_str.split():
                    if len(wd) > 1:
                       text_str2.append(wd)

                text_str = ' '.join(text_str2)

                if len(text_str) >= 3:
                    if traceid == '925':
                       act_para = {'para_id': prop['id'], 'description': preprocess_text(text_str), 'Type': prop_dict['Type']}
                    else:
                       act_para = {'para_id': prop[pId][0], 'description': preprocess_text(text_str), 'Type': prop_dict['Type']}

                    if prop_dict[aId] in para_DB:
                        para_DB[prop_dict[aId]].append(act_para)
                    else:
                        para_DB[prop_dict[aId]] = [act_para]

    return node_DB, para_DB


def get_para_name(para_id, para_DB):
    for action_id, act_tup_list in para_DB.items():
        for act_tup in act_tup_list:
            if act_tup['para_id'] == para_id:
                return act_tup['description']
    return 'null'


def get_parameters_content(trace_id, ext_KB, para_DB):
    query = 'g.V().has("TraceId", "'+trace_id+'").hasLabel("parameter").repeat(out()).emit().valueMap(true)'
    result_list = execute_query(query, client)

    para_dom = {}
    for res1 in result_list:
        for res2 in res1:

            text_str = res2['description'][0].lower()
            if "|" in text_str:
                text_str = text_str.split("|")[0]
            text_str = replace_punctuations(text_str)

            if 'parameterId' not in res2:
                print(res2)

            if get_para_name(res2['parameterId'][0], para_DB) == 'date':
                for para_val in ext_KB['date']:
                    if res2['parameterId'][0] in para_dom:
                        para_dom[res2['parameterId'][0]].add(para_val.lower())
                    else:
                        para_dom[res2['parameterId'][0]] = {para_val.lower()}
            else:
                if res2['parameterId'][0] in para_dom:
                    para_dom[res2['parameterId'][0]].add(' '.join(text_str.split()))
                else:
                    para_dom[res2['parameterId'][0]] = {' '.join(text_str.split())}

    return para_dom


def get_neighbors(node_id, client, edge_type=""):
    if edge_type == "":
        query = 'g.V().has("id","'+node_id+'").outE()'
    else:
        query = 'g.V().has("id","'+node_id+'").outE().haslabel("'+edge_type+'")'
    result_list = execute_query(query, client)

    neighbors = set()
    for result in result_list:
        for prop in result:
            neighbors.add(prop['inV'])

    return neighbors


def execute_query(query, client):
    callback = client.submitAsync(query)
    return callback.result()


def get_property(prop, prop_set, node_type):
    prop_dict = {}
    for prop_type in prop_set[node_type]:
        if prop_type in prop:
            prop_dict[prop_type] = prop[prop_type][0]
    return prop_dict


def get_all_activity_content(trace_id, node_DB):
    content_DB = {}

    # html titles
    get_html_titles(trace_id, content_DB, node_DB)

    # html descriptions...
    get_html_descriptions(trace_id, content_DB, node_DB)

    # get schemaorg info ...
    get_schema_org(trace_id, content_DB, node_DB)

    # get opengraph info ....
    get_open_graph_info(trace_id, content_DB, node_DB)

    return content_DB


def get_html_titles(trace_id, content_DB, node_DB):
    query = 'g.V().has("TraceId", "' + trace_id + '").hasLabel("knowledge").out().has("description", "html").' \
                                                  'out().has("description","title").repeat(out()).emit().valueMap(true)'
    result_list = execute_query(query, client)

    if result_list is not None:
        populate_content_DB(content_DB, node_DB, result_list, "page_description")


def get_html_descriptions(trace_id, content_DB, node_DB):
    query = 'g.V().has("TraceId", "' + trace_id + '").hasLabel("knowledge").out().has("description", "html").' \
                                                  'out().has("description","description").repeat(out()).emit().valueMap(true)'
    result_list = execute_query(query, client)

    if result_list is not None:
       populate_content_DB(content_DB, node_DB, result_list, "page_content")


def get_open_graph_info(trace_id, content_DB, node_DB):
    query = 'g.V().has("TraceId", "' + trace_id + '").hasLabel("knowledge").out().has("description", "opengraph").' \
                                                  'out().has("description", "title").repeat(out()).emit().valueMap(true)'
    result_list = execute_query(query, client)

    if result_list is not None:
        populate_content_DB(content_DB, node_DB, result_list, "page_description")

    query = 'g.V().has("TraceId", "' + trace_id + '").hasLabel("knowledge").out().has("description", "opengraph").' \
                                                  'out().has("description", "type").repeat(out()).emit().valueMap(true)'
    result_list = execute_query(query, client)
    if result_list is not None:
        populate_content_DB(content_DB, node_DB, result_list, "page_description")

    query = 'g.V().has("TraceId", "' + trace_id + '").hasLabel("knowledge").out().has("description", "opengraph").' \
                                                  'out().has("description", "description").repeat(out()).emit().valueMap(true)'
    result_list = execute_query(query, client)
    if result_list is not None:
        populate_content_DB(content_DB, node_DB, result_list, "page_content")


def get_immutables(trace_id, content_DB, node_DB):
    query = 'g.V().has("TraceId", "' + trace_id + '").hasLabel("knowledge").out().has("description", "immutables").' \
                                                  'repeat(out()).emit().valueMap(true)'
    result_list = execute_query(query, client)

    if result_list is not None:
         populate_content_DB(content_DB, node_DB, result_list, "page_content")


def get_ocr(trace_id, content_DB, node_DB):
    query = 'g.V().has("TraceId", "' + trace_id + '").hasLabel("knowledge").out().has("description", "ocr").' \
                                                  'repeat(out()).emit().valueMap(true)'
    result_list = execute_query(query, client)

    if result_list is not None:
        populate_content_DB(content_DB, node_DB, result_list, "page_content")


def get_schema_org(trace_id, content_DB, node_DB):
    query = 'g.V().has("TraceId", "' + trace_id + '").hasLabel("knowledge").out().has("description", "schemaorg").' \
                                                  'repeat(out()).emit().valueMap(true)'
    result_list = execute_query(query, client)
    if result_list is not None:
         populate_content_DB(content_DB, node_DB, result_list, "page_content")


def populate_content_DB(content_DB, node_DB, result_list, text_tag, freq_thresh = 0.5, freq_flag=False):
    for res1 in result_list:
        for res2 in res1:
            if res2['description'][0] in NO_text_set:
                continue
            else:
                text_str = res2['description'][0].lower()
                if '###' in text_str:
                    text_str = text_str.split("###")[0]

                if 'icon/ic_' in text_str or 'icon /' in text_str:
                    text_str = text_str.split("ic_")[1]
                    text_str = text_str.replace("_", " ")

                if "|" in text_str:
                    text_str = text_str.split("|")[0]

                text_str2 = []
                for wd in text_str.split():
                    if len(wd) > 1:
                       text_str2.append(wd)

                text_str = ' '.join(text_str2)

                if len(text_str) <= 3:
                    continue

                snapshot_count = int(node_DB['activity'][res2['activityId'][0]]['snapshotCount'])
                if freq_flag and (snapshot_count == 0 or (text_tag == 'page_content' and int(res2['frequency'][0]) < 1)):
                    continue
                else:
                    processed_text = replace_punctuations(text_str)
                    if processed_text != '':
                        if res2['activityId'][0] in content_DB:
                            if text_tag in content_DB[res2['activityId'][0]]:
                                content_DB[res2['activityId'][0]][text_tag].add(processed_text)
                            else:
                                content_DB[res2['activityId'][0]][text_tag] = {processed_text}
                        else:
                            content_DB[res2['activityId'][0]] = {text_tag: {processed_text}}

    for activity_id in content_DB:
        if text_tag in content_DB[activity_id]:
            remove_phrase = set()

            for phrase in content_DB[activity_id][text_tag]:
                if len(phrase.split()) > 1:
                    jointphrase = phrase.replace(' ', '')
                    if jointphrase in content_DB[activity_id][text_tag]:
                        remove_phrase.add(jointphrase)

            filtered_phrase_set = content_DB[activity_id][text_tag].difference(remove_phrase)

            content_DB[activity_id][text_tag] = set()
            for phrase in filtered_phrase_set:
                content_DB[activity_id][text_tag].add(preprocess_text(phrase))
