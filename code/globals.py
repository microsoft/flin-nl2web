# Uncased ...

bert_vocab_file = '....'  # add the path of BERT Tokenizer Vocab file here ..

max_seq_len = {'query': 30, 'word': 12, 'activity_desc': 20, 'activity_content_phrase': 10,
                   'action_name': 5, 'para_name': 5, 'para_val': 7,
                   'bert_seq_len_act': 85, 'bert_seq_len_para':28,
                   'bert_query': 48, 'bert_action_name': 10, 'bert_para_name': 10, 'bert_para_val': 14,
                   'bert_para_names_str': 100, 'bert_query_para_name': 60}

max_no_dict = {'phrases_per_activity_desc': 5, 'phrases_per_activity_content': 15,
               'max_no_para_per_action': 12, 'max_no_prev_action': 4, 'domain_size_per_para': 50}
