__author__ = 'jiachiliu'

from numpy.linalg import inv
import numpy as np


class LinearRegression(object):
    """docstring for LinearRegression"""

    def __init__(self):
        self.coeff = None

    def fit(self, train, target):
        self.coeff = inv(train.T.dot(train)).dot(train.T).dot(target)
        return self

    def predict(self, test):
        return test.dot(self.coeff)
