__author__ = 'jiachiliu'

from numpy.linalg import inv
import numpy as np
import sys


class LinearRegression(object):
    """docstring for LinearRegression"""

    def __init__(self):
        self.weights = None

    def fit(self, train, target):
        self.weights = inv(train.T.dot(train)).dot(train.T).dot(target)
        return self

    def predict(self, test):
        return test.dot(self.weights)


class GradientDescendingRegression(LinearRegression):
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, train, target, alpha=0.00001, max_loop=1500):
        m, n = train.shape
        self.weights = np.ones(n)
        for k in range(max_loop):
            predict = self.predict(train)
            error = predict - target
            self.weights -= alpha * train.T.dot(error)

        return self


class StochasticGradientDescendingRegression(LinearRegression):
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, train, target, alpha=0.0001, max_loop=130):
        m, n = train.shape
        self.weights = np.zeros(n)
        for k in range(max_loop):
            self.print_progress(k, max_loop)
            for t in range(m):
                data_point = train[t]
                error = self.predict(data_point) - target[t]
                self.weights -= alpha * error * data_point
        return self

    @staticmethod
    def print_progress(cur, max_loop):
        sys.stdout.write('\r Loop: %d of %d' % (cur + 1, max_loop))
        sys.stdout.flush()


class LogisticGradientDescendingRegression(StochasticGradientDescendingRegression):
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, train, target, alpha=0.0001, max_loop=130):
        m, n = train.shape
        self.weights = np.zeros(n)
        for k in range(max_loop):
            self.print_progress(k, max_loop)
            for t in xrange(m):
                data_point = train[t]
                predict = self.predict(data_point)
                error = predict - target[t]
                self.weights -= alpha * error * predict * (1.0 - predict) * data_point
        return self

    @staticmethod
    def sigmoid(vals):
        return 1.0 / (1 + np.exp(-vals))

    def predict(self, test):
        return self.sigmoid(test.dot(self.weights))

    @staticmethod
    def convert_to_binary(vals, threshold=0.5):
        return map(lambda v: 1 if v >= threshold else 0, vals)




