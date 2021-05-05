import random
import string

import numpy as np
np.random.seed(1234)
random.seed(1234)


def replace_punctuations(s, default_char=''):
    ''' punctuation removal '''

    for c in string.punctuation:
        if c == '-':
            s = s.replace(c, ' ')
        if c not in {':', '$', '@', '.', '/', '&', '%', '<', '>', '_', '\'', '-'}:
            s = s.replace(c, default_char)
    return s


def get_activity_id(node_DB, activity_name):
    for activity_id in node_DB['activity']:
        if node_DB['activity'][activity_id]['ActivityName'] == activity_name:
            return activity_id
    return '-'


def get_parameter_list(action_str, para_DB, action_DB):

     for action_id in action_DB[action_str.split('#')[0]]:
         para_list = set()
         action_name = ''

         for para in para_DB[action_id]:
             if para['Type'] == 'ACTION':
                 action_name = para['description']

             if para['Type'] == 'INPUT' or para['Type'] == 'TEXT':
                 para_list.add(para['description'])

         if action_name == action_str.split('#')[1]:
             return list(para_list)

     return []


def get_parameter_value(action_name, para_name, p_DB):

    if action_name in p_DB and para_name in p_DB[action_name]:
        para_list = p_DB[action_name][para_name]
        para_type = p_DB[action_name][para_name][0][2]
        for i in range(3):
          random.shuffle(para_list)

        if len(para_list[0][1]) > 0:
            return para_list[0][0], np.random.choice(list(para_list[0][1]), 1)[0], para_type
        else:
            return para_list[0][0], para_list[0][0], para_type
    return 'NULL', 'NULL', 'NULL'
