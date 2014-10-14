__author__ = 'jiachiliu'

from nulearn.dataset import load_spambase
from nulearn.bayes import *
from nulearn.cross_validation import *
from nulearn.validation import *
import matplotlib.pyplot as plt

roc_data = []

def naive_bayes(c, on_train=False):
    train, target = load_spambase()
    train, target = shuffle(train, target)

    k = 10
    train_size = len(train)
    test_index_generator = k_fold_cross_validation(train_size, k)

    fold = 1

    overall_acc = 0
    overall_error = 0
    overall_auc = 0

    overall_train_acc = 0
    overall_train_error = 0

    for start, end in test_index_generator:
        k_fold_train = np.vstack((train[range(0, start)], train[range(end, train_size)]))
        test = train[range(start, end)]

        train_target = np.append(target[range(0, start)], target[range(end, train_size)])
        test_target = target[range(start, end)]

        cf = get_classifier(c)
        raw_predicts = cf.fit(k_fold_train, train_target).predict(test)
        predicts = cf.predict_class(raw_predicts)

        if on_train:
            raw_train_predicts = cf.predict(k_fold_train)
            train_predicts = cf.predict_class(raw_train_predicts)
            print '=============Fold %s Train==================' % fold
            cm = confusion_matrix(train_target, train_predicts)
            print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
            er, acc, fpr, tpr = confusion_matrix_analysis(cm)
            print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)
            overall_train_acc += acc
            overall_train_error += er


        print '=============Fold %s Test==================' % fold
        cm = confusion_matrix(test_target, predicts)
        print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
        er, acc, fpr, tpr = confusion_matrix_analysis(cm)
        print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)

        roc = ROC(test_target, raw_predicts, 0, "NaN")
        recs = roc.create_roc_records()
        roc.plot_roc_data(recs)
        auc = roc.auc()
        print 'AUC: %s' % auc

        if fold == 1:
            roc_data.append(roc.points)

        overall_acc += acc
        overall_error += er
        overall_auc += auc
        fold += 1

    print '--------------- Result-------------------'
    if on_train:
        print 'Overall Accuracy: %s, Overall Error: %s' % (overall_train_acc/k, overall_train_error/k)
    print 'Overall Accuracy: %s, Overall Error: %s, Overall AUC: %s\n' % (overall_acc/k, overall_error/k, overall_auc/k)


def get_classifier(c):
    if c == 'bernoulli':
        return BernoulliNaiveBayes()
    if c == 'gaussian':
        return GaussianNaiveBayes()
    if c == 'histogram':
        return HistogramNaiveBayes()
    if c == 'GDA':
        return GDA()
    return NBinsHistogramNaiveBayes(c)


def gda():
    naive_bayes('GDA', True)


def gaussian_naive_bayes():
    naive_bayes('gaussian')


def bernoulli_naive_bayes():
    naive_bayes('bernoulli')


def histogram_naive_bayes():
    naive_bayes('histogram')


def n_bins_histogram_naive_bayes():
    naive_bayes(9)


def gaussian_naive_bayes1():
    train, target = load_spambase()
    train, test, train_target, test_target = train_test_shuffle_split(train, target, len(train) / 10)
    cf = GaussianNaiveBayes()
    predicts = cf.fit(train, train_target).predict_class(test)
    cm = confusion_matrix(test_target, predicts)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)


def plot():
    for points, m, c in zip(roc_data,'oooo', 'rgby'):
        x = points[:, 1]
        y = points[:, 0]
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.scatter(x, y, marker=m, c=c)
    plt.show()

if __name__ == '__main__':
    bernoulli_naive_bayes()
    gaussian_naive_bayes()
    histogram_naive_bayes()
    n_bins_histogram_naive_bayes()
    # gda()
    plot()

