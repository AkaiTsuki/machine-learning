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