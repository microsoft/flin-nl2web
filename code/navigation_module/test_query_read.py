import random
from string import punctuation
random.seed(1234)


def read_gold_file(path, q_type='all', ext=False):
    data_file = open(path, "r")

    q_DB = {}
    i = 0

    for line in data_file.readlines():
        i += 1

        data_fields = line.strip().split(',')
        print(str(data_fields) + '---------')

        if data_fields[0] == '' or data_fields[1] =='':
            continue

        query_type = data_fields[0].split("D:")[1].strip()
        if q_type == 'E' and query_type != q_type:
            continue
        if q_type == 'I' and query_type != q_type:
            continue

        test_query = data_fields[1].split("Q:")[1].lower().strip()
        gold_path_1 = data_fields[2].split("P1:")[1].strip()

        if test_query == '':
            continue

        if gold_path_1 == '':
            continue

        if '||' in gold_path_1:
            gold_path_list = gold_path_1.split('||')
        else:
            gold_path_list = [gold_path_1]

        ground_truth_list = []
        for i in range(0, len(gold_path_list), 1):
            gold_path = gold_path_list[i]

            action_desc_name = gold_path.split("{")[0]
            if i == 0:
                action_desc_name = action_desc_name.split(";")[1]
            st_node = action_desc_name.split("->")[0]

            para_str = gold_path.split("{")[1].replace("}", '').replace("\'", '').split(';')

            para_dict = {}
            for para_name_val in para_str:
                if "=" in para_name_val:
                    if ext:
                        para_name = para_name_val.split("=")[0].replace("\"", '').strip()
                        para_val = para_name_val.split("=")[1].replace("\"", '').strip()
                        para_dict[para_name] = para_val
                    else:
                        para_name = para_name_val.split("=")[0].replace("\"", '').strip()
                        para_val = para_name_val.split("=")[1].replace("\"", '').strip().split('|')[0].strip()
                        para_type = para_name_val.split("=")[1].replace("\"", '').strip().split('|')[1].strip()
                        para_dict[para_name] = para_val

            print(st_node, action_desc_name, para_dict)
            ground_truth_list.append((st_node, action_desc_name, para_dict))

        q_DB[test_query.strip().strip(punctuation)] = ground_truth_list

        if i % 100 == 0:
            print(i)
    data_file.close()
    print("# test_queries: ", len(list(q_DB.keys())))
    return q_DB


if __name__ == '__main__':
    q_DB = read_gold_file('../result_958_pred_gold.csv')

