__author__ = 'jiachiliu'

from nulearn.preprocessing import append_new_column
from nulearn.preprocessing import normalize
from nulearn.dataset import load_boston_house
from nulearn.dataset import load_spambase
from nulearn.linear_model import LinearRegression
from nulearn.validation import mae
from nulearn.validation import mse
from nulearn.validation import rmse
from nulearn import cross_validation
import numpy as np
from nulearn.validation import confusion_matrix
from nulearn.validation import confusion_matrix_analysis
import sys


def housing():
    train, train_target, test, test_target = load_boston_house()

    normalize_columns = [0, 1, 2, 6, 7, 9, 10, 11, 12]
    normalize(train, normalize_columns)
    normalize(test, normalize_columns)
    train = append_new_column(train, 1.0, 0)
    test = append_new_column(test, 1.0, 0)

    lr = LinearRegression()
    lr.fit(train, train_target)

    print '=============Train Data Result============'
    predict = lr.predict(train)
    print "mse: ", mse(predict, train_target), " rmse: ", rmse(predict, train_target), " mae: ", mae(predict,
                                                                                                     train_target)
    print '=============Test Data Result============'
    predict = lr.predict(test)
    print "mse: ", mse(predict, test_target), " rmse: ", rmse(predict, test_target), " mae: ", mae(predict, test_target)


def spam():
    train, target = load_spambase()

    normalize_columns = [55, 56]
    normalize(train, normalize_columns)
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

        cf = LinearRegression()
        cf = cf.fit(k_fold_train, train_target)

        print '=============Train Data Result============'
        predict_train = cf.predict(k_fold_train)
        cm = confusion_matrix(train_target, predict_train)

        er, acc, fpr, tpr = confusion_matrix_analysis(cm)
        train_accuracy += acc
        print "mse: ", mse(predict_train, train_target), " rmse: ", rmse(predict_train, train_target), " mae: ", mae(
            predict_train,
            train_target)
        train_mse += mse(predict_train, train_target)

        print '=============Test Data Result============'
        predict_test = cf.predict(test)
        cm = confusion_matrix(test_target, predict_test)

        er, acc, fpr, tpr = confusion_matrix_analysis(cm)
        test_accuracy += acc
        fold += 1
        print "mse: ", mse(predict_test, test_target), " rmse: ", rmse(predict_test, test_target), " mae: ", mae(
            predict_test,
            test_target)
        test_mse += mse(predict_test, test_target)

    print "Average train mse: %f, average test mse: %f" % (1.0*train_mse / fold, 1.0*test_mse / fold)


def main():
    if sys.argv[1] == "housing":
        housing()
    elif sys.argv[1] == "spam":
        spam()
    else:
        print "Invalid dataset please use [housing] or [spam]."


if __name__ == '__main__':
    main()