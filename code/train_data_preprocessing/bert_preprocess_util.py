from bert import tokenization
from code.globals import max_seq_len, bert_vocab_file
import re


def create_tokenizer(vocab_file, do_lower_case=False):
    return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer(bert_vocab_file)


def get_vectorized_bert_input_phrase(input_sentence, bert_seq_len):
    # encode query ... first input
    tokens = ['[CLS]']
    sent_tokens = tokenizer.tokenize(input_sentence)
    if len(sent_tokens) > bert_seq_len - 2:
        sent_tokens = sent_tokens[:bert_seq_len - 2]
    tokens.extend(sent_tokens)
    tokens.append('[SEP]')

    segment_ids = [0] * len(tokens)

    # convert to ids ...
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero Mask till seq_length
    zero_mask = [0] * (bert_seq_len - len(tokens))

    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    assert len(input_ids) == len(input_mask) == len(segment_ids) == bert_seq_len

    return (input_ids, input_mask, segment_ids)


def isSublist(pattern, mylist):
    if len(pattern) > len(mylist):
        return (0, 0)
    for i in range(len(mylist) - len(pattern) + 1):
        if mylist[i:i+len(pattern)] == pattern:
            return (i, i+len(pattern)-1)
    return (0, 0)


def get_bert_input_query_para_name(q_phrase, para_name, para_val_sample, bert_seq_len):
    mention_tokens = tokenizer.tokenize(para_val_sample)

    # encode para name ... # first input
    tokens = ['[CLS]']
    para_name_tokens = tokenizer.tokenize(para_name)
    if len(para_name_tokens) > max_seq_len['bert_para_name'] - 1:
        para_name_tokens = para_name_tokens[:max_seq_len['bert_para_name'] - 1]
    tokens.extend(para_name_tokens)
    tokens.append('[SEP]')
    segment_ids = [0] * len(tokens)

    # encode query ... # second input
    q_tokens = tokenizer.tokenize(q_phrase)
    if len(q_tokens) > max_seq_len['bert_query'] - 1:
        q_tokens = q_tokens[:max_seq_len['bert_query'] - 1]
    tokens.extend(q_tokens)
    tokens.append('[SEP]')

    segment_ids.extend([1] * (len(q_tokens)+1))

    # extract mention start and end ids ...
    gold_label_ids = isSublist(mention_tokens, tokens)
    if gold_label_ids[0]!=0 and gold_label_ids[1]!=0:
        assert len(mention_tokens) == (gold_label_ids[1]-gold_label_ids[0]+1)

    # convert to ids ...
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero Mask till seq_length
    padding = [0] * (bert_seq_len - len(input_ids))

    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == len(input_mask) == len(segment_ids) == bert_seq_len

    return (input_ids, input_mask, segment_ids), gold_label_ids


def get_bert_input_query_para_name_test(q_phrase, para_name, bert_seq_len):
    # encode para name ... # first input
    tokens = ['[CLS]']
    para_name_tokens = tokenizer.tokenize(para_name)
    if len(para_name_tokens) > max_seq_len['bert_para_name'] - 1:
        para_name_tokens = para_name_tokens[:max_seq_len['bert_para_name'] - 1]
    tokens.extend(para_name_tokens)
    tokens.append('[SEP]')
    segment_ids = [0] * len(tokens)

    # encode query ... # second input
    q_tokens = tokenizer.tokenize(q_phrase)
    if len(q_tokens) > max_seq_len['bert_query'] - 1:
        q_tokens = q_tokens[:max_seq_len['bert_query'] - 1]
    tokens.extend(q_tokens)
    tokens.append('[SEP]')

    q_token_dict = []
    for wd in q_phrase.split():
        q_token_dict.append((wd, tokenizer.tokenize(wd)))

    segment_ids.extend([1] * (len(q_tokens)+1))

    # convert to ids ...
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero Mask till seq_length
    padding = [0] * (bert_seq_len - len(input_ids))

    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == len(input_mask) == len(segment_ids) == bert_seq_len

    return (input_ids, input_mask, segment_ids), tokens, q_token_dict
