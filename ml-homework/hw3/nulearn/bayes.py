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
                log_liklihood = 0.0
                for f in range(test.shape[1]):
                    # print "index, feature, mean, var: %s %s %s %s" % (f, t[f],self.get_class_conditional_mean(l, f), self.get_class_conditional_var(l, f))
                    g = self.gaussian_on_ln(t[f], self.get_class_conditional_mean(l, f),
                                            self.get_class_conditional_var(l, f))
                    log_liklihood += g
                res.append(log_liklihood + np.log(self.priors[l]))
            predicts.append(res)
        return predicts

    def predict_class(self, test):
        predicts = self.predict(test)
        # print predicts
        return np.array(map(lambda p: 1.0 if p[0] <= p[1] else 0.0, predicts))

    @staticmethod
    def gaussian(f, m, v):
        v2 = ((f - m) * (f - m)) / (2.0 * v)
        v1 = np.exp(-v2)
        v3 = v1 / np.sqrt(2.0 * v * np.pi)
        return v3

    @staticmethod
    def gaussian_on_ln(f, m, v):
        v1 = np.log(1.0 / np.sqrt(2.0 * v * np.pi))
        v2 = -((f - m) ** 2) / (2.0 * v)
        return v1 + v2

    def get_class_conditional_mean(self, label, feature):
        return self.conditional_mean[label][feature]

    def get_class_conditional_var(self, label, feature):
        return self.conditional_var[label][feature]


class HistogramNaiveBayes:
    def __init__(self):
        self.overall_mean = None
        self.spam_mean = None
        self.non_spam_mean = None
        self.priors = {}
        self.bins = []
        self.likelihoods = {}
        self.labels = None

    def fit(self, train, target):
        self.setup_bins(train, target)
        self.labels = np.unique(target)

        for l in self.labels:
            tuples = train[target == l]
            self.calculate_likelihoods(tuples, l)
        return self

    def calculate_likelihoods(self, data, label):
        label_count = len(data)
        self.likelihoods[label] = []
        possible_value_count = len(self.bins[0]) - 1
        for f in range(data.shape[1]):
            feature_values = data[:, f]
            bin = self.bins[f]
            bin_count = [1] * possible_value_count
            for val in feature_values:
                bin_count[self.get_bin_index(val, bin)] += 1

            bin_likelihoods = []
            for c in bin_count:
                bin_likelihoods.append(1.0 * c / (label_count + possible_value_count))
            self.likelihoods[label].append(bin_likelihoods)

    def get_bin_index(self, val, bin):
        for i in range(len(bin) - 1):
            if i == 0 and bin[i] == val:
                return i
            if bin[i] < val <= bin[i + 1]:
                return i
        return -1

    def setup_bins(self, train, target):
        self.overall_mean = self.get_mean_vector(train)
        spams = train[target == 1]
        non_spams = train[target == 0]
        self.priors[1] = 1.0 * len(spams) / len(train)
        self.priors[0] = 1.0 * len(non_spams) / len(train)

        self.spam_mean = self.get_mean_vector(spams)
        self.non_spam_mean = self.get_mean_vector(non_spams)

        min_val_vector = [train[:, f].min() for f in range(train.shape[1])]
        max_val_vector = [train[:, f].max() for f in range(train.shape[1])]

        for f in range(train.shape[1]):
            min_value = min_val_vector[f]
            max_value = max_val_vector[f]
            spam_mean_value = self.spam_mean[f]
            non_spam_mean_value = self.non_spam_mean[f]
            mean_value = self.overall_mean[f]
            bin = [min_value, max_value, spam_mean_value, non_spam_mean_value, mean_value]
            bin = sorted(bin)
            self.bins.append(bin)

    def predict(self, test):
        predicts = []

        for t in test:
            res = []
            for l in self.labels:
                likelihoods = self.likelihoods[l]
                posterior = 1.0
                for f in range(test.shape[1]):
                    posterior *= likelihoods[f][self.get_bin_index(t[f], self.bins[f])]
                res.append(posterior * self.priors[l])
            predicts.append(res)

        return predicts

    def predict_class(self, test):
        predicts = self.predict(test)
        return np.array(map(lambda p: 1.0 if p[0] <= p[1] else 0.0, predicts))

    @staticmethod
    def get_mean_vector(data):
        return np.array([data[:, f].mean() for f in range(data.shape[1])])
