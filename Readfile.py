from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class Readfile:

    def __init__(self, path1, path2, index):
        self.features_name = np.zeros((1, 1))
        self.features_value = np.zeros((1, 1))
        self.features_class = np.zeros((1,))
        self.data_set = np.zeros((1, 1))
        self.text_neg = open(path1, 'r').readlines()
        self.text_pos = open(path2, 'r').readlines()
        self.text_all = []

        self.neg_size = len(self.text_neg)
        self.pos_size = len(self.text_pos)
        self.neg_value = 0
        self.pos_value = 1
        self._combine_data()
        self._extract_features(index)
        self._add_class()

    def _combine_data(self):
        for line in self.text_neg:
            print(line)
            self.text_all.append(line)
        for line in self.text_pos:
            print(line)
            self.text_all.append(line)

    def _extract_features(self, index):
        if index == 1:
            vectorizer = CountVectorizer()
        else:
            vectorizer = CountVectorizer(ngram_range=(1, index), token_pattern=r'\b\w+\b', min_df=1)
        x = vectorizer.fit_transform(self.text_all)
        self.features_name = vectorizer.get_feature_names()
        self.features_value = x.toarray()
        print('The bag of words contains %d-gram features' % index)

    def _add_class(self):

        neg_col = np.full((self.neg_size, 1), self.neg_value)
        pos_col = np.full((self.pos_size, 1), self.pos_value)

        self.features_class = np.vstack((neg_col, pos_col))
        self.data_set = np.append(self.features_value, self.features_class, axis=1)


