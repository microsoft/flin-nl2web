from code.dataset_genaration.load_additional_resources import load_external_resources
import numpy as np, random
np.random.seed(1234)
random.seed(1234)


def read_annotated_train_file(trace_id, args, over_sampling = False):
    data_file = open(args['data_path'] + 'annotation/' + str(trace_id) + '_train.csv', "r")

    q_DB = {}
    i = 0
    max_no_patt = 0

    for line in data_file.readlines():
        i += 1
        # if i == 1:
        #     continue

        data_fields = line.strip().split(',')

        if len(data_fields) != 4:
            print(line, ' ...... parse error!')
            print(data_fields)
            input()
            continue

        sd_nodes = data_fields[0].split("SD:")[1].strip()
        path_len = data_fields[1].split("PL:")[1].strip()
        path = data_fields[2].split("PATH:")[1].strip()
        q_patterns = data_fields[3].split("QP:")[1].strip()

        if int(path_len) > 1:
            continue

        if q_patterns == '':
            continue

        for q_pattern in q_patterns.split(";"):
            if q_pattern != '':
                if path in q_DB:
                    q_DB[path].add(q_pattern)
                else:
                    q_DB[path] = {q_pattern}

        if max_no_patt < len(q_DB[path]):
            max_no_patt = len(q_DB[path])

        if i % 100 == 0:
            print(i)

    data_file.close()
    print("# paths: ", len(list(q_DB.keys())))
    print('max # templates per path', max_no_patt)

    if over_sampling:
        for path, pattern_set in q_DB.items():
            extended_pattern_list = []
            extended_pattern_list.extend(list(pattern_set))
            if len(pattern_set) < max_no_patt:
                oversampled_patterns = np.random.choice(list(pattern_set), max_no_patt - len(pattern_set), replace=True)
                extended_pattern_list.extend(list(oversampled_patterns))
            assert len(extended_pattern_list) == max_no_patt
            q_DB[path] = extended_pattern_list
        print("query template Oversampling done!")
    return q_DB


def read_annotated_para_file(trace_id, args, ext_KB):
    para_file = open(args['data_path'] + 'annotation/' + str(trace_id) + '_para.csv', "r")

    p_DB = {}
    j = 0
    max_no_para_phrases = 0

    for line2 in para_file.readlines():
        j += 1
        # if j == 1:
        #     continue

        data_fields = line2.strip().split(',')

        if len(data_fields) != 5:
            print(data_fields)
            input()

        act_name = data_fields[0].strip()
        para_name = data_fields[1].lower().strip()
        para_val_fixed = data_fields[2].lower().strip()
        paraphrases = data_fields[3].lower().strip()
        para_type = data_fields[4].lower().strip()

        if para_type == 'open' or para_type == 'text':
            para_type = 0
        else:
            para_type = 1

        if act_name == '' or para_val_fixed == '':
            print(line2, ' ...... parse error!')
            print(data_fields)
            input()
            continue

        if para_name == '-':
            para_name = act_name.split("#")[1].lower().strip()

        if para_name.strip() in {'reservation date', 'date', 'check out', 'check in',
                                 'check out date', 'check in date',
                                 'check-out', 'check-in'}:
            ext_para_list = []
            for para_val_ext, paraphrase_set in ext_KB['date'].items():
                ext_para_list.append((para_val_ext, paraphrase_set, 1))

            if act_name in p_DB:
               if para_name not in p_DB[act_name]:
                   p_DB[act_name][para_name] = ext_para_list
            else:
               p_DB[act_name] = {para_name: ext_para_list}
            print(para_name + '.....date data added')
        else:
            paraphrase_set = set()
            for phrase in paraphrases.split(";"):
                if phrase != '':
                    paraphrase_set.add(phrase.lower().strip())

            if len(paraphrase_set) > max_no_para_phrases:
                max_no_para_phrases = len(paraphrase_set)

            if act_name in p_DB:
                if para_name in p_DB[act_name]:
                    para_list = p_DB[act_name][para_name]
                    para_list.append((para_val_fixed, paraphrase_set, para_type))
                    p_DB[act_name][para_name] = para_list
                else:
                    p_DB[act_name][para_name] = [(para_val_fixed, paraphrase_set, para_type)]
            else:
                p_DB[act_name] = {para_name: [(para_val_fixed, paraphrase_set, para_type)]}

        if j % 100 == 0:
            print(j)

    print('max # para phrases', max_no_para_phrases)
    return p_DB


def read_annotated_dataset(trace_id, args):
    '''
        Read the human annotated _train.csv and _para.csv files for labeled NL utterance instantiation
    '''

    ext_KB = load_external_resources()

    q_DB = read_annotated_train_file(trace_id, args)
    p_DB = read_annotated_para_file(trace_id, args, ext_KB)

    print('FILE read complete!')
    print('# path-Query:', len(q_DB))
    print('# lines in parameter file:', len(p_DB))
    return q_DB, p_DB
