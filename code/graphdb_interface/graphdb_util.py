from code.dataset_genaration.data_gen_util import replace_punctuations


def preprocess_text(phrase):
    phrase = replace_punctuations(phrase)
    return phrase.lower().strip()


def get_action_description(action_DB, action_id):
    for action_desc in action_DB:
        if action_id in action_DB[action_desc]:
            return action_desc
    return ''


def get_action_name(para_DB, action_id):
    for para in para_DB[action_id]:
        if para['Type'] == 'ACTION':
            return para['description']
    return '-'


def get_action_names_from_desc(act_desc, action_DB, para_DB):
    action_ids = action_DB[act_desc]

    act_names = set()
    for action_id in action_ids:
        act_names.add(get_action_name(para_DB, action_id))
    return act_names


def get_action_content(para_dom, node_DB, para_DB, activity_id, act_type):
    act_content = set()

    activity_name = node_DB['activity'][activity_id]['ActivityName']

    act_set = set()
    for act_desc in node_DB['action']:
        if act_type == 'out' and act_desc.split("->")[0] == activity_name:
            act_set.add(act_desc)
        if act_type == 'in' and act_desc.split("->")[1] == activity_name:
            act_set.add(act_desc)

    for act_desc in act_set:
        for action_id in node_DB['action'][act_desc]:
            for para in para_DB[action_id]:
                if para['para_id'] in para_dom:
                    for phrase in para_dom[para['para_id']]:
                         act_content.add(phrase.lower())
                act_content.add(para['description'].lower())
    return act_content