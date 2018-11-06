import numpy as np

class Models:

    def __init__(self):
        self.name = "Naive Bayes Laplace Smoothing"

    @staticmethod
    def add_one(data):
        return data + 1

    @staticmethod
    def binary_feature(data):
        data = np.where(data > 0, 1, data)
        return data
