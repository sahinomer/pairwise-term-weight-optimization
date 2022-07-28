import os

import json
import numpy as np

from utils import extract_phrase


class Query:
    def __init__(self, qid, query, phrase_length):
        self.qid = qid
        self.query = query
        self.query_terms = extract_phrase(self.query, max_phrase_length=phrase_length)
        self._relevant_feature_list = list()
        self._irrelevant_feature_list = list()
        self._weights = None

    def add_sample(self, relevant, irrelevant):
        self._relevant_feature_list.append(relevant)
        self._irrelevant_feature_list.append(irrelevant)

    @property
    def sequence_length(self):
        return len(self._relevant_feature_list[0])

    @property
    def relevant_features(self):
        return np.array(self._relevant_feature_list)

    @property
    def irrelevant_features(self):
        return np.array(self._irrelevant_feature_list)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def to_file(self, output):
        extension = os.path.splitext(output.name)[1]
        if extension == '.json':
            output.write(self.to_json() + '\n')
        else:
            output.write(self.to_tsv() + '\n')

    def to_tsv(self, precision='.10f'):

        term_list = []
        for i, term in enumerate(self.query_terms):

            window_size = len(term.split(' '))
            if window_size == 1:
                term_list.append(f'{term}^{self._weights[i]:{precision}}')
            else:
                term_list.append(f'"{term}"^{self._weights[i]:{precision}}')

        return self.qid + '\t' + ' '.join(term_list)

    def to_json(self):
        term_weight = dict()
        for i, term in enumerate(self.query_terms):
            term_weight[term] = float(self._weights[i])

        return json.dumps({'qid': self.qid, 'query': self.query, 'term_weight': term_weight}, ensure_ascii=False)
