import abc
import math
import string
import time
from collections import OrderedDict

import numpy as np
from fuzzywuzzy import process
from nltk.stem import WordNetLemmatizer
from scipy import spatial

try:
    import ujson as json
except ImportError:
    import json


class Matcher(abc.ABC):

    def __init__(self, resource_file_list=None, cache_size=1000000):
        self.resource_files = resource_file_list
        self.score_cache = dict()
        self.CACHE_SIZE = cache_size

    @abc.abstractmethod
    def _match_score(self, src_str, tgt_str):
        pass

    @staticmethod
    def sim_func(vector_s, vector_t):
        return 1 - spatial.distance.cosine(vector_s, vector_t)

    @staticmethod
    def get_score(src_v, tgt_v):
        if any(tgt_v) and any(src_v):
            node_score = Matcher.sim_func(src_v, tgt_v)
        else:
            node_score = 0
        if math.isnan(node_score):
            node_score = 0
        return node_score

    def get_match_score(self, src_str, tgt_str):
        str_pair = (src_str, tgt_str)
        # if it's in the cache don't invoke the matcher
        if str_pair in self.score_cache:
            return self.score_cache[str_pair]
        else:
            if src_str and tgt_str:
                score = self._match_score(src_str, tgt_str)
            else:
                score = 0
            self._may_clear_cache(self.score_cache)
            self.score_cache[str_pair] = score
            return score

    def _may_clear_cache(self, cache):
        if len(cache) > self.CACHE_SIZE:
            cache.clear()


class LexicalMatcher(Matcher):
    # TODO: add 'jaro_winkler', 'smith_waterman' , but need to improve speed
    # NAMES = ('lexical', 'jaccard', 'cosine', 'levenshtein', 'jaro_winkler', 'smith_waterman')
    NAMES = ('lexical', 'jaccard', 'cosine', 'levenshtein_distance', 'jaro_winkler')

    def __init__(self, name='lexical'):
        super(LexicalMatcher, self).__init__()

        if name not in LexicalMatcher.NAMES:
            raise ValueError('{} not supported!'.format(name))
        # print('initializing {} matcher...'.format(name))
        start = time.time()
        self.name = name
        self.lemmatizer = WordNetLemmatizer()
        self.translator = str.maketrans('', '', string.punctuation)
        # self.tf_idf_vec = TfidfVectorizer(min_df=1)
        # import textdistance
        import jellyfish

        # self.td_matchers = {n: getattr(textdistance, n).normalized_similarity for n in LexicalMatcher.NAMES[3:]}
        self.td_matchers = {n: getattr(jellyfish, n) for n in LexicalMatcher.NAMES[3:]}
        print('{} matcher initialized, took {:4.4f} s'.format(name, time.time() - start))

    def _match_score(self, src_str, tgt_str):
        if self.name == 'lexical':
            scores = OrderedDict()
            for n in LexicalMatcher.NAMES[1:]:
                scores[n] = self.one_match_score(src_str, tgt_str, n)
            return sum(scores.values())
        else:
            return self.one_match_score(src_str, tgt_str, self.name)

    def one_match_score(self, src_str, tgt_str, name):
        # import textdistance
        # td_matcher = getattr(textdistance, self.name)
        # return td_matcher.normalized_similarity(src_str, tgt_str)

        if name == 'jaccard' or name == 'cosine':
            # stemming is not a good idea
            # ps = PorterStemmer()
            # stem1 = ps.stem(str1)
            stra = self.lemmatizer.lemmatize(src_str.lower().translate(self.translator))
            strb = self.lemmatizer.lemmatize(tgt_str.lower().translate(self.translator))
            a = set(stra.split())
            b = set(strb.split())
            c = a.intersection(b)
            if name == 'jaccard':
                return float(len(c)) / (len(a) + len(b) - len(c))
            else:
                norm1 = len(a) ** (1. / 2)
                norm2 = len(b) ** (1. / 2)
                if not norm1 * norm2:
                    return 0
                else:
                    return len(c) / (norm1 * norm2)
        # elif name == 'cosine':
        #     tf_idf = self.tf_idf_vec.fit_transform([src_str, tgt_str])
        #     return (tf_idf * tf_idf.T).A[0, 1]
        else:
            score = self.td_matchers[name](src_str, tgt_str)
            if name == 'levenshtein_distance':
                score = 1 - score / max(len(src_str), len(tgt_str))
            return score


matcher_dict = dict()
NAMES = ['universal', 'hybrid', 'semantic', 'lexical',
         'match_pyramid', 'cdssm', 'word2vec',
         'jaccard', 'levenshtein_distance', 'cosine', 'jaro_winkler']


def get_matcher(name):
        # get_matcher('word2vec'))
        # if name == 'universal':
        #   return UniversalMatcher()
        if name == 'lexical':
            return LexicalMatcher('levenshtein_distance')
        #matcher_dict[name] = matcher_
        return matcher_lex


def get_partial_match_score(cand_wd, q_phrase):
    max_match = 0.0
    for q_wd in q_phrase.lower().split():
        if cand_wd.startswith(q_wd) or cand_wd.endswith(q_wd):
            sim = (len(q_wd) * 1.0) / len(cand_wd)
            if sim >= 0.5 and sim > max_match:
                max_match = sim
            if 0.5 > sim >= 0.12 and q_wd.isdigit() and sim > max_match:
                max_match = sim
    return max_match


def get_custom_match_score(q_phrase, cand_phrase):
    cand_wd_set = cand_phrase.lower().split()

    score_list = {}
    for wd in cand_wd_set:
        match_score = get_partial_match_score(wd, q_phrase)
        #print(wd, match_score)
        score_list[wd] = match_score
    if len(score_list) > 0:
        return 1.0 - np.mean(list(score_list.values()))
    else:
        return 1.0


def get_fuzzy_matching(source_phrase, target_value):
    sim_score = process.extractOne(source_phrase, [target_value])
    return (100.0 - sim_score[1]) / 100.0


matcher_lex = get_matcher('lexical')


def get_ext_matching_score(source_phrase, target_value):
    fuzzy_score = get_fuzzy_matching(source_phrase, target_value)
    lexical_score = 1.0 - matcher_lex.get_match_score(source_phrase, target_value)
    custom_score = get_custom_match_score(source_phrase, target_value)
    return [custom_score * 0.7 + lexical_score * 0.1 + fuzzy_score * 0.2]
