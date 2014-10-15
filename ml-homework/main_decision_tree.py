__author__ = 'jiachiliu'

from nulearn import cross_validation
from nulearn import tree
from nulearn.validation import mae
from nulearn.validation import mse
from nulearn.validation import rmse
from nulearn.validation import confusion_matrix
from nulearn.validation import confusion_matrix_analysis
from nulearn.dataset import load_spambase
from nulearn.dataset import load_boston_house
from nulearn.tree import print_tree
import numpy as np
import logging
import sys


def decision_tree_all_data():
    train, target = load_spambase()
    cf = tree.DecisionTree()
    cf = cf.fit(train, target, 5)
    print_tree(cf.root)
    predicts = cf.predict(train)
    cm = confusion_matrix(target,predicts)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)


def decision_tree():
    train, target = load_spambase()

    # 10 fold cross validation
    train_size = len(train)
    k = 10
    test_index_generator = cross_validation.k_fold_cross_validation(train_size, k)
    fold = 0
    train_accuracy = 0
    test_accuracy = 0
    train_mse = 0
    test_mse = 0

    for start, end in test_index_generator:
        train_left = train[range(0, start)]
        train_right = train[range(end, train_size)]
        k_fold_train = np.vstack((train_left, train_right))
        test = train[range(start, end)]

        target_left = target[range(0, start)]
        target_right = target[range(end, train_size)]
        train_target = np.append(target_left, target_right)
        test_target = target[range(start, end)]

        cf = tree.DecisionTree()
        cf = cf.fit(k_fold_train, train_target, 5)
        print "=========Tree=============="
        print_tree(cf.root)

        print '=============Train Data Result============'
        predict_train = cf.predict(k_fold_train)
        cm = confusion_matrix(train_target, predict_train)
        print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
        er, acc, fpr, tpr = confusion_matrix_analysis(cm)
        print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)
        train_accuracy += acc
        print "mse: ", mse(predict_train, train_target), " rmse: ", rmse(predict_train, train_target), " mae: ", mae(predict_train,
                                                                                                     train_target)
        train_mse += mse(predict_train, train_target)

        print '=============Test Data Result============'
        predict_test = cf.predict(test)
        cm = confusion_matrix(test_target, predict_test)
        print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
        er, acc, fpr, tpr = confusion_matrix_analysis(cm)
        print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)
        test_accuracy += acc
        print "mse: ", mse(predict_test, test_target), " rmse: ", rmse(predict_test, test_target), " mae: ", mae(predict_test,
                                                                                                     test_target)
        test_mse += mse(predict_test, test_target)

        fold += 1

    print "Average train acc: %f, average test acc: %f" % (train_accuracy / fold, test_accuracy / fold)
    print "Average train mse: %f, average test mse: %f" % (train_mse / fold, test_mse / fold)


def regression_tree():
    print "=========Start Train=============="
    train, train_target, test, test_target = load_boston_house()
    print len(train), len(test)
    classifier = tree.RegressionTree()
    classifier = classifier.fit(train, train_target, 2, -1)

    print "=========Finish Train=============="
    print "=========Tree=============="
    print_tree(classifier.root)

    print '=============Train Data Result============'
    predict = classifier.predict(train)
    print "mse: ", mse(predict, train_target), " rmse: ", rmse(predict, train_target), " mae: ", mae(predict,
                                                                                                     train_target)

    print '=============Test Data Result============'
    predict = classifier.predict(test)
    print "mse: ", mse(predict, test_target), " rmse: ", rmse(predict, test_target), " mae: ", mae(predict, test_target)

def main1():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s", level=logging.DEBUG)
    decision_tree_all_data()

def main():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s", level=logging.DEBUG)
    if sys.argv[1] == "housing":
        regression_tree()
    elif sys.argv[1] == "spam":
        decision_tree()
    else:
        print "Invalid dataset, please use [housing] or [spam]."

if __name__ == '__main__':
    main()