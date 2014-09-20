__author__ = 'jiachiliu'

from nulearn.preprocessing import append_new_column
from nulearn.preprocessing import normalize
from nulearn.dataset import load_boston_house
from nulearn.dataset import load_spambase
from nulearn.linear_model import *
from nulearn.validation import *
from nulearn import cross_validation
import numpy as np
import sys


def housing():
    train, train_target, test, test_target = load_boston_house()

    data = np.vstack((train, test))
    normalize(data)
    train = data[:len(train)]
    test = data[len(train):]
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


def spam1():
    train, target = load_spambase()

    #normalize_columns = [55, 56]
    normalize(train)
    train = append_new_column(train, 1.0, 0)

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

        cf = LogisticGradientDescendingRegression()
        cf = cf.fit(k_fold_train, train_target)

        print '=============Train Data Result============'
        predict_train = cf.predict(k_fold_train)
        predict_train = cf.convert_to_binary(predict_train)

        cm = confusion_matrix(train_target, predict_train)
        print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
        er, acc, fpr, tpr = confusion_matrix_analysis(cm)
        print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)
        train_accuracy += acc

        print "mse: ", mse(predict_train, train_target), " rmse: ", rmse(predict_train, train_target), " mae: ", mae(
            predict_train,
            train_target)
        train_mse += mse(predict_train, train_target)

        print '=============Test Data Result============'
        predict_test = cf.predict(test)
        predict_test = cf.convert_to_binary(predict_test)
        cm = confusion_matrix(test_target, predict_test)
        print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
        er, acc, fpr, tpr = confusion_matrix_analysis(cm)
        print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)
        test_accuracy += acc
        fold += 1
        print "mse: ", mse(predict_test, test_target), " rmse: ", rmse(predict_test, test_target), " mae: ", mae(
            predict_test,
            test_target)
        test_mse += mse(predict_test, test_target)

    print "Average train acc: %f, average test acc: %f" % (train_accuracy / fold, test_accuracy / fold)
    print "Average train mse: %f, average test mse: %f" % (1.0*train_mse / fold, 1.0*test_mse / fold)


def spam_logistic(train, test, train_target, test_target, step, loop):
    cf = LogisticGradientDescendingRegression()
    cf = cf.fit(train, train_target, step, loop)
    print '=============Train Data Result============'
    predict_train = cf.predict(train)
    predict_train = cf.convert_to_binary(predict_train)

    cm = confusion_matrix(train_target, predict_train)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)

    print '=============Test Data Result============'
    predict_test = cf.predict(test)
    roc(test_target, predict_test, "ROC Logistic Regression")
    predict_test = cf.convert_to_binary(predict_test)
    cm = confusion_matrix(test_target, predict_test)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)


def spam_linear(train, test, train_target, test_target, step, loop):
    cf = StochasticGradientDescendingRegression()
    cf = cf.fit(train, train_target, step, loop)
    print '=============Train Data Result============'
    predict_train = cf.predict(train)
    predict_train = LogisticGradientDescendingRegression.convert_to_binary(predict_train)

    cm = confusion_matrix(train_target, predict_train)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)

    print '=============Test Data Result============'
    predict_test = cf.predict(test)
    roc(test_target, predict_test, "ROC Linear Regression")
    predict_test = LogisticGradientDescendingRegression.convert_to_binary(predict_test)
    cm = confusion_matrix(test_target, predict_test)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)


def spam(step, loop):
    train, target = load_spambase()
    normalize(train)
    train = append_new_column(train, 1.0, 0)

    train, test, train_target, test_target = cross_validation.train_test_shuffle_split(train, target, len(train)/10)
    # Logistic Regression
    spam_logistic(train, test, train_target, test_target, step, loop)
    # Linear Regression
    spam_linear(train, test, train_target, test_target, step, loop)


def main():
    if sys.argv[1] == "housing":
        housing()
    elif sys.argv[1] == "spam":
        spam(0.001, 100)
    else:
        print "Invalid dataset please use [housing] or [spam]."


if __name__ == '__main__':
    main()