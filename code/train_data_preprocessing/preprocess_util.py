import nltk
import random
import spacy
import string

import numpy as np
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from spacy.tokens import Doc

np.random.seed(1234)
random.seed(1234)

stopWords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

char_vocab_to_id = {'char_PAD': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11,
                    'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23,
                    'x': 24, 'y': 25, 'z': 26,
                    '$': 27, '-': 28, ':': 29, '@': 30, '.': 31, '/': 32, '\'': 33, '&': 44, '%': 45, '<': 46, '>': 47, '_': 48,
                    '0': 34, '1': 35, '2': 36, '3': 37, '4': 38, '5': 39, '6': 40, '7': 41, '8': 42, '9': 43}

ent_vocab_to_id = {'ent_PAD': 0, 'GPE': 1, 'LOC': 2, 'DATE': 3, 'TIME': 4, 'MONEY': 5, 'ORDINAL':6, 'CARDINAL': 7}


def replace_punctuations(s, default_char=''):
    ''' punctuation removal '''

    for c in string.punctuation:
        if c == '-':
            s = s.replace(c, ' ')
        if c not in {':', '$', '@', '.', '/', '\'', '&', '%', '<', '>'}:
            s = s.replace(c, default_char)
    return s


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def get_vectorized_char_seq(phrase, char_vocab_to_id, q_len, q_wd_len):
    q_char_vec = []

    for wd in phrase.split():
        wd_vec = []
        for char in wd:
           if char in char_vocab_to_id:
               wd_vec.append(char_vocab_to_id[char])
           else:
               wd_vec.append(0)

        if len(wd_vec) >= q_wd_len:
            wd_vec = wd_vec[:q_wd_len]
        else:
            wd_vec = pad_arr_seq(wd_vec, q_wd_len, 0)
        q_char_vec.append(wd_vec)

    if len(q_char_vec) >= q_len:
        return q_char_vec[:q_len]
    else:
        return pad_arr_seq(q_char_vec, q_len, [0] * q_wd_len)


def get_gold_labels_tagger(q_phrase, para_val_sample, max_seq_len):
    # q_phrase = ' '.join(q_phrase.split())
    para_val_sample = ' '.join(para_val_sample.split())
    q_word_list = q_phrase.split()
    para_val_sample_word_list = para_val_sample.split()

    label_vec = [0] * len(q_phrase.split())
    index_list = []
    for wd_id, q_word in enumerate(q_word_list):
        if q_word == para_val_sample_word_list[0]:
             if ' '.join(q_word_list[wd_id:]).startswith(para_val_sample):
                 for j in range(len(para_val_sample_word_list)):
                     index_list.append(wd_id+j)

    for pos_id in index_list:
        label_vec[pos_id] = 1
    assert len(label_vec) == len(q_phrase.split())

    if len(label_vec) >= max_seq_len:
        return label_vec[:max_seq_len], len(q_word_list)
    else:
        return pad_arr_seq(label_vec, max_seq_len, 0), len(q_word_list)


def get_vectorized_entity_tags(phrase, ent_vocab_to_id, q_len):
    q_ent_tag_vec = []

    phrase = phrase.strip()
    doc = nlp(phrase)
    word_tags = []

    for i in range(len(doc)):
        word_tags.append((doc[i].text, doc[i].ent_iob_, doc[i].ent_type_))

        if doc[i].ent_type_ in ent_vocab_to_id:
            q_ent_tag_vec.append(ent_vocab_to_id[doc[i].ent_type_])
        else:
            q_ent_tag_vec.append(ent_vocab_to_id['ent_PAD'])

    if len(q_ent_tag_vec) >= q_len:
        return q_ent_tag_vec[:q_len]
    else:
        return pad_arr_seq(q_ent_tag_vec, q_len, 0)


def get_query_n_grams(q_phrase, max_n=3, min_n=1):
    q_words = q_phrase.lower().split()

    pos_tag_dict = {tup[0]:tup[1] for tup in nltk.pos_tag(q_words)}
    exclueded_pos_set = { 'VB', 'VBD', 'VBG', 'VBZ'}

    q_uni_bigram_phrases = set()
    for n_gr in range(min_n, max_n+1, 1):
        n_gram_list = list(ngrams(q_words, n_gr))

        for tup in n_gram_list:
            n_gram_phrase = ' '.join([wd for wd in list(tup) if wd not in stopWords
                                      and pos_tag_dict[wd] not in exclueded_pos_set])

            if n_gram_phrase != '':
                q_uni_bigram_phrases.add(n_gram_phrase.strip())
    return q_uni_bigram_phrases


def pad_arr_seq(curr_seq, max_len, padding_seq):

    for i in range(max_len-len(curr_seq)):
         curr_seq.append(padding_seq)
    assert len(curr_seq) == max_len
    return curr_seq


def get_activity_id(node_DB, activity_name):
    for activity_id in node_DB['activity']:
        if node_DB['activity'][activity_id]['ActivityName'] == activity_name:
            return activity_id
    return '-'


def preprocess_text(phrase):
    phrase = replace_punctuations(phrase)

    if len(phrase) < 3:
        return ''
    token_list = []

    for wd in phrase.split():
        # if wd in stopWords:
        #     continue

        if not wd.isdigit():
             token_list.append(lemmatizer.lemmatize(wd))
        else:
             token_list.append(wd)

    return ' '.join(token_list)


def has_partial_match(wd, cand_wd_set):
    cand_wd = ' '.join(cand_wd_set)
    if cand_wd.startswith(wd) or cand_wd.endswith(wd):
        sim = (len(wd) * 1.0) / len(cand_wd)
        #print(wd, sim, cand_wd)
        if 0.5 > sim >= 0.12 and wd.isdigit():
            return 1, True
        if sim >= 0.5:
            return 2, True
    return 0, False


def get_match_vec(q_phrase, cand_phrase, max_q_len):
    '''

    :param q_phrase:
    :param cand_phrase:
    :param max_q_len:
    :return:
    '''
    q_match_vec = []

    cand_wd_set = cand_phrase.lower().split()

    for wd in q_phrase.lower().split():
        if wd in cand_wd_set:
            q_match_vec.append(3)
        else:
            match_id, is_match = has_partial_match(wd, cand_wd_set)
            q_match_vec.append(match_id)

    if len(q_match_vec) >= max_q_len:
        return q_match_vec[:max_q_len]
    else:
        return pad_arr_seq(q_match_vec, max_q_len, 0)


def get_vectorized_phrase(phrase, vocab_to_id, max_seq_len):
    phrase_vec = []

    for wd in phrase.split():
        if wd in vocab_to_id:
            phrase_vec.append(vocab_to_id[wd])
        else:
            phrase_vec.append(0)

    if len(phrase_vec) >= max_seq_len:
        return phrase_vec[:max_seq_len]
    else:
        return pad_arr_seq(phrase_vec, max_seq_len, 0)


def extract_noun_phrases(sentence):
    doc = nlp(sentence)
    noun_phrases = set()

    exclude_set = set()
    for token in doc:
        if token.pos_ in {'PRON'}:
           exclude_set.add(token.text)

    for chunk in doc.noun_chunks:
        noun_phrases.add(chunk.text)

    noun_phrases = noun_phrases.difference(exclude_set)
    return noun_phrases


def get_candidate_query_phrases(sentence):
    noun_P = extract_noun_phrases(sentence)
    q_n_grams = get_query_n_grams(sentence)
    return q_n_grams.union(noun_P)


if __name__ == '__main__':
    print(get_gold_labels_tagger('new york hotels for for 10 people','for 10 people' , 10))