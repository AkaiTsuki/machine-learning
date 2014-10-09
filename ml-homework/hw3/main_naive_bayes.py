__author__ = 'jiachiliu'

from nulearn.dataset import load_spambase
from nulearn.bayes import BernoulliNaiveBayes
from nulearn.cross_validation import *
from nulearn.validation import *


def bernoulliNB():
    train, target = load_spambase()
    train, target = shuffle(train, target)

    k = 10
    train_size = len(train)
    test_index_generator = k_fold_cross_validation(train_size, k)

    fold = 1

    overall_acc = 0
    overall_error = 0

    for start, end in test_index_generator:
        k_fold_train = np.vstack((train[range(0, start)], train[range(end, train_size)]))
        test = train[range(start, end)]

        train_target = np.append(target[range(0, start)], target[range(end, train_size)])
        test_target = target[range(start, end)]

        cf = BernoulliNaiveBayes()
        predicts = cf.fit(k_fold_train, train_target).predict_class(test)

        print '=============Fold %s==================' % fold
        cm = confusion_matrix(test_target, predicts)
        print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
        er, acc, fpr, tpr = confusion_matrix_analysis(cm)
        print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)

        overall_acc += acc
        overall_error += er
        fold += 1

    print '--------------- Result-------------------'
    print 'Overall Accuracy: %s, Overall Error: %s\n' % (overall_acc/k, overall_error/k)


if __name__ == '__main__':
    bernoulliNB()