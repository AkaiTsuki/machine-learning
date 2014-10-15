__author__ = 'jiachiliu'

import numpy as np
import logging

def print_tree(root):
    if root:
        print root
        print_tree(root.left)
        print_tree(root.right)


class TreeNode:
    """docstring for TreeNode"""

    def __init__(self, left, right, f, val, level):
        self.left = left
        self.right = right
        self.feature = f
        self.val = val
        self.level = level

    def __str__(self):
        return "|" + "----" * self.level + "[level: " + str(self.level) + " feature: " + str(
            self.feature) + ", value: " + str(self.val) + " ]"


class BaseTree:
    def __init__(self):
        # total number of features
        self.total_features = 0
        # total number of train data point
        self.total_train = 0
        # max level for the decision tree
        self.max_level = 0
        # the root of decision tree
        self.root = None
        # the minimum number of data point in each node
        self.min_data_in_node = 30

    def fit(self, train, target, max_level=5, min_data_in_node=30):
        self.total_train, self.total_features = train.shape
        self.max_level = max_level
        self.min_data_in_node = min_data_in_node
        self.root = self.build_tree(train, target, range(self.total_features), 0)
        return self

    def build_tree(self, train, target, features, level):
        logging.debug("Train data size: %d", len(train))

        if self.is_all_same_label(target):
            return TreeNode(None, None, self.total_features, target[0], level)
        if len(features) == 0:
            return TreeNode(None, None, self.total_features, self.majority_vote(target), level)
        if level >= self.max_level:
            return TreeNode(None, None, self.total_features, self.majority_vote(target), level)
        if len(train) <= self.min_data_in_node:
            return TreeNode(None, None, self.total_features, self.majority_vote(target), level)

        (f, s) = self.find_best_split_feature(train, target, features)
        logging.debug("(best_feature, split_value) = (%s, %s)", f, s)
        if f is None or s is None:
            logging.warn("Selected feature's split value is None: (%s, %s)", f, s)
            return TreeNode(None, None, self.total_features, self.majority_vote(target), level)

        left_train = train[train[:, f] <= s]
        right_train = train[train[:, f] > s]
        left_target = target[train[:, f] <= s]
        right_target = target[train[:, f] > s]

        logging.debug("data split[left = %d, right = %d]", len(left_target), len(right_target))

        if len(left_train) == 0:
            left_tree = TreeNode(None, None, self.total_features, self.majority_vote(target), level)
        else:
            left_tree = self.build_tree(left_train, left_target, features, level + 1)

        if len(right_train) == 0:
            right_tree = TreeNode(None, None, self.total_features, self.majority_vote(target), level)
        else:
            right_tree = self.build_tree(right_train, right_target, features, level + 1)

        return TreeNode(left_tree, right_tree, f, s, level)

    def find_best_split_feature(self, train, target, features):
        best_feature = None
        best_split_value = None
        max_score = -float('inf')
        score_d = self.measure(target)

        for f in features:
            (split, score) = self.find_best_split_on_feature(train[:, f], target, score_d)
            # logging.debug("(feature, split, score): (%s, %s, %s)", f, split, score)
            if max_score <= score:
                max_score = score
                best_feature = f
                best_split_value = split
        return best_feature, best_split_value

    def find_best_split_on_feature(self, feature_vals, target, parent_score):
        max_score = -float('inf')
        best_split_val = None
        sorted_features, sorted_target = self.get_sorted_feature_and_target(feature_vals, target)

        for i in range(1, len(sorted_features)):
            if sorted_features[i] == sorted_features[i - 1]:
                continue
            split = (sorted_features[i] + sorted_features[i - 1]) / 2.0
            left = sorted_target[:i]
            right = sorted_target[i:]
            score = self.measure_on_children(left, right, sorted_target, parent_score)
            if max_score <= score:
                max_score = score
                best_split_val = split

        return best_split_val, max_score

    def predict_single_data(self, x):
        node = self.root
        while node is not None and node.feature != len(x):
            if x[node.feature] <= node.val:
                node = node.left
            else:
                node = node.right
        return node.val

    def predict(self, test):
        predict = []
        for t in test:
            predict.append(self.predict_single_data(t))
        return np.array(predict)

    @staticmethod
    def get_sorted_feature_and_target(features, target):
        sorted_index = features.argsort()
        sorted_target = target[sorted_index]
        sorted_features = features[sorted_index]
        return sorted_features, sorted_target

    def measure(self, target):
        pass

    def measure_on_children(self, left, right, parent_target, parent_score):
        pass

    def is_all_same_label(self, target):
        pass

    def majority_vote(self, target):
        pass


class DecisionTree(BaseTree):
    def __init__(self):
        BaseTree.__init__(self)

    def measure_on_children(self, left, right, target, parent_score):
        score_left = (1.0 * len(left) / len(target)) * self.measure(left)
        score_right = (1.0 * len(right) / len(target)) * self.measure(right)
        return parent_score - (score_left + score_right)

    def measure(self, target):
        info = 0.0
        total = len(target) * 1.0

        if total == 0:
            return info

        for val in np.unique(target):
            d_i = len(target[target == val])
            info += -(d_i / total) * np.log2(d_i / total)

        return info

    def majority_vote(self, target):
        max_count = 0
        l = target[0]

        for v in np.unique(target):
            c = len(target[target == v])
            if max_count < c:
                max_count = c
                l = v
        return l

    def is_all_same_label(self, target):
        l = target[0]
        for v in target:
            if l != v:
                return False
        return True


class RegressionTree(BaseTree):
    def __init__(self):
        BaseTree.__init__(self)

    def measure(self, target):
        sse = 0.0
        if len(target) == 0:
            return sse

        mean = target.mean()
        for e in target:
            sse += (e - mean) ** 2
        return sse

    def measure_on_children(self, left, right, target, parent_score):
        return parent_score - (self.measure(left) + self.measure(right))

    def majority_vote(self, target):
        return target.mean()

    def is_all_same_label(self, target):
        return False