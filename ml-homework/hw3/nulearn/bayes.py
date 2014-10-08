__author__ = 'jiachiliu'

import numpy as np


class BernoulliNaiveBayes:
    def __init__(self, overall_mean):
        self.overall_mean = overall_mean
        self.probabilities = {}
        self.labels = None
        self.labels_prob = []

    def fit(self, train, target):
        # find unique class labels
        labels = np.unique(target).tolist()
        self.labels = labels

        # find how many tuples for each label
        # and split train tuples based on label
        label_tuples = {}
        label_counts = {}
        for l in labels:
            label_tuples[l] = train[target == l]
            label_counts[l] = len(label_tuples[l])
            self.labels_prob.append(1.0 * label_counts[l] / len(target))
            # calculate probabilities for each feature
            self.calculate_probabilities(label_tuples[l], label_counts[l], l)
        return self

    def calculate_probabilities(self, data, label_count, label):
        self.probabilities[label] = []
        for f in range(data.shape[1]):
            feature_vals = data[:, f]
            less_than_mean = len(feature_vals[feature_vals <= self.overall_mean[f]])
            greater_than_mean = len(feature_vals[feature_vals > self.overall_mean[f]])
            pr_less_than_mean = 1.0 * (less_than_mean + 1) / (2 + label_count)
            pr_greater_than_mean = 1.0 * (greater_than_mean + 1) / (2 + label_count)
            self.probabilities[label].append((pr_less_than_mean, pr_greater_than_mean))

    def predict(self, test):
        predicts = []

        for t in test:
            res = []
            for l in self.labels:
                probs = self.probabilities[l]
                prob = 1.0
                for f in range(test.shape[1]):
                    if t[f] <= self.overall_mean[f]:
                        prob *= probs[f][0]
                    else:
                        prob *= probs[f][1]
                res.append(prob * self.labels_prob[int(l)])
            predicts.append(res)

        return predicts

    def predict_class(self, test):
        predicts = self.predict(test)
        return np.array(map(lambda p: 1.0 if p[0] <= p[1] else 0.0, predicts))


    def get_overall_mean(self, train):
        return [train[:, f].mean() for f in range(train.shape[1])]