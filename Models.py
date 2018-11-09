import numpy as np


class Models:

    def __init__(self):
        self.name = "Naive Bayes Laplace Smoothing"

    @staticmethod
    def add_one(data):
        return data + 1

    @staticmethod
    def binary_feature(data):
        data = np.where(data > 0, int(1), data)
        return data

    @staticmethod
    def combine_bayes_logistic(data):
        row, col = data.shape
        X = data[:, :col-1]
        y = data[:, col-1:col]
        p = np.zeros((1, col-1), dtype=int)
        q = np.zeros((1, col-1), dtype=int)
        for id in range(row):
            if y[id] == 1:
                p = np.add(p, X[id, :])
            else:
                q = np.add(q, X[id, :])
        p = p + 1
        q = q + 1
        p = p/np.sum(p, axis=1)
        q = q/np.sum(q, axis=1)
        ratio = np.log2(np.divide(p, q))
        return ratio

    @staticmethod
    def process_vector_with_ratio(data, ratio):
        row, col = data.shape
        X = data[:, :col-1]
        y = data[:, col-1:col]
        for id in range(row):
            X[id, :] = np.multiply(X[id, :], ratio)
        return np.append(X, y, axis=1)
