__author__ = 'jiachiliu'

from nulearn.preprocessing import append_new_column
from nulearn.preprocessing import normalize
from nulearn.dataset import load_boston_house
from nulearn.dataset import load_spambase
from nulearn.linear_model import *
from nulearn.tree import *
from nulearn.validation import *
from nulearn import cross_validation
import numpy as np
import sys


def housing():
    train, train_target, test, test_target = load_boston_house()

    scaler = normalize(train)
    scaler.scale_test(test)

    train = append_new_column(train, 1.0, 0)
    test = append_new_column(test, 1.0, 0)

    lr = StochasticGradientDescendingRegression()
    # lr = GradientDescendingRegression()
    lr.fit(train, train_target, 0.0001, 500)

    print '=============Train Data Result============'
    predict = lr.predict(train)
    print "mse: ", mse(predict, train_target), " rmse: ", rmse(predict, train_target), " mae: ", mae(predict,
                                                                                                     train_target)
    print '=============Test Data Result============'
    predict = lr.predict(test)
    print "mse: ", mse(predict, test_target), " rmse: ", rmse(predict, test_target), " mae: ", mae(predict, test_target)


def spam_decision_tree(train, test, train_target, test_target):
    cf = DecisionTree()
    cf = cf.fit(train, train_target, 5)
    test_and_evaluate(cf, train, test, train_target, test_target, 'Spam Decision Tree', False)


def spam_logistic(train, test, train_target, test_target, step, loop, converge):
    cf = LogisticGradientDescendingRegression()
    cf = cf.fit(train, train_target, step, loop, converge)
    test_and_evaluate(cf, train, test, train_target, test_target,
                      'Spam Logistic Regression - Stochastic Gradient Descending')


def spam_linear(train, test, train_target, test_target, step, loop, converge):
    cf = StochasticGradientDescendingRegression()
    cf = cf.fit(train, train_target, step, loop, converge)
    test_and_evaluate(cf, train, test, train_target, test_target,
                      'Spam Linear Regression - Stochastic Gradient Descending')


def spam_normal_equation(train, test, train_target, test_target):
    cf = LinearRegression()
    cf = cf.fit(train, train_target)
    test_and_evaluate(cf, train, test, train_target, test_target, 'Spam Normal Equation')


def test_and_evaluate(cf, train, test, train_target, test_target, title, use_roc=True):
    print '=============Train Data Result============'
    predict_train = cf.predict(train)
    predict_train = LogisticGradientDescendingRegression.convert_to_binary(predict_train)

    cm = confusion_matrix(train_target, predict_train)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)

    print '=============Test Data Result============'
    predict_test = cf.predict(test)
    if use_roc:
        roc(test_target, predict_test, title)
    predict_test = LogisticGradientDescendingRegression.convert_to_binary(predict_test)
    cm = confusion_matrix(test_target, predict_test)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)


def spam(step, loop, converge):
    train, target = load_spambase()

    train, test, train_target, test_target = cross_validation.train_test_shuffle_split(train, target, len(train) / 10)
    scaler = normalize(train)
    train = append_new_column(train, 1.0, 0)
    scaler.scale_test(test)
    test = append_new_column(test, 1.0, 0)

    print '\n============== Logistic Regression - Stochastic Gradient Descending==============='
    spam_logistic(train, test, train_target, test_target, step, loop, converge)

    print '\n============== Linear Regression - Stochastic Gradient Descending ==============='
    spam_linear(train, test, train_target, test_target, step, loop, converge)

    print '\n============== Linear Regression - Normal Equation==============='
    spam_normal_equation(train, test, train_target, test_target)

    print '\n============== Decision Tree ===================================='
    spam_decision_tree(train, test, train_target, test_target)


def main():
    if sys.argv[1] == "housing":
        housing()
    elif sys.argv[1] == "spam":
        spam(0.001, 100, 0.0001)
    else:
        print "Invalid dataset please use [housing] or [spam]."


if __name__ == '__main__':
    main()