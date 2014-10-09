__author__ = 'jiachiliu'

import numpy as np


class BernoulliNaiveBayes:
    def __init__(self):
        self.overall_mean = None
        self.likelihoods = {}
        self.labels = None
        self.priors = {}

    def setup(self, train, target):
        self.overall_mean = self.get_overall_mean(train)
        # find unique class labels
        self.labels = np.unique(target)

    def fit(self, train, target):
        # get mean vector for each features
        # get all distinct labels
        self.setup(train, target)

        # find how many tuples for each label
        # and split train tuples based on label
        for l in self.labels:
            # find all tuples under this class label
            tuples = train[target == l]
            # get the count
            counts = len(tuples)
            # calculate priors
            self.priors[l] = 1.0 * counts / len(target)
            # calculate likelihoods for each feature on current label
            self.calculate_likelihoods(tuples, counts, l)
        return self

    def calculate_likelihoods(self, data, label_count, label):
        self.likelihoods[label] = []
        for f in range(data.shape[1]):
            feature_values = data[:, f]
            less_than_mean = len(feature_values[feature_values <= self.overall_mean[f]])
            greater_than_mean = len(feature_values[feature_values > self.overall_mean[f]])
            pr_less_than_mean = 1.0 * (less_than_mean + 1) / (2 + label_count)
            pr_greater_than_mean = 1.0 * (greater_than_mean + 1) / (2 + label_count)
            self.likelihoods[label].append((pr_less_than_mean, pr_greater_than_mean))

    def predict(self, test):
        predicts = []

        for t in test:
            res = []
            for l in self.labels:
                likelihoods = self.likelihoods[l]
                posterior = 1.0
                for f in range(test.shape[1]):
                    posterior *= likelihoods[f][self.get_likelihood_index(t[f], self.overall_mean[f])]
                res.append(posterior * self.priors[l])
            predicts.append(res)

        return predicts

    def predict_class(self, test):
        predicts = self.predict(test)
        return np.array(map(lambda p: 1.0 if p[0] <= p[1] else 0.0, predicts))

    @staticmethod
    def get_likelihood_index(f, mean):
        return 0 if f <= mean else 1

    @staticmethod
    def get_overall_mean(train):
        return [train[:, f].mean() for f in range(train.shape[1])]


class GaussianNaiveBayes:
    def __init__(self):
        # the mean vector for all features
        self.overall_mean = None
        # the variance vector for all features
        self.overall_var = None
        # class conditional mean
        self.conditional_mean = {}
        # class conditional var
        self.conditional_var = {}
        # all labels
        self.labels = None
        self.priors = {}

    def setup(self, train, target):
        self.overall_mean = self.get_mean_vector(train)
        self.overall_var = self.get_var_vector(train)
        self.labels = np.unique(target)

    @staticmethod
    def get_mean_vector(data):
        return np.array([data[:, f].mean() for f in range(data.shape[1])])

    @staticmethod
    def get_var_vector(data):
        return np.array([data[:, f].var() for f in range(data.shape[1])])

    def fit(self, train, target):
        self.setup(train, target)
        n = len(target)
        p = 1.0 * n / (n + 2)
        for l in self.labels:
            tuples = train[target == l]
            self.priors[l] = 1.0 * len(tuples) / n
            self.conditional_mean[l] = self.get_mean_vector(tuples)
            self.conditional_var[l] = p * self.get_var_vector(tuples) + (1 - p) * self.overall_var
        return self

    def predict(self, test):
        predicts = []
        for t in test:
            res = []
            for l in self.labels:
                liklihood = 1.0
                for f in range(test.shape[1]):
                    #print "index, feature, mean, var: %s %s %s %s" % (f, t[f],self.get_class_conditional_mean(l, f), self.get_class_conditional_var(l, f))
                    g = self.gaussian(t[f], self.get_class_conditional_mean(l, f), self.get_class_conditional_var(l, f))
                    liklihood *= g
                res.append(liklihood * self.priors[l])
            predicts.append(res)
        return predicts

    def predict_class(self, test):
        predicts = self.predict(test)
        #print predicts
        return np.array(map(lambda p: 1.0 if p[0] <= p[1] else 0.0, predicts))

    @staticmethod
    def gaussian(f, m, v):
        v2 = ((f - m) * (f - m)) / (2.0 * v)
        v1 = np.exp(-v2)
        v3 = v1 / np.sqrt(2.0 * v * np.pi)
        return v3

    def get_class_conditional_mean(self, label, feature):
        return self.conditional_mean[label][feature]

    def get_class_conditional_var(self, label, feature):
        return self.conditional_var[label][feature]