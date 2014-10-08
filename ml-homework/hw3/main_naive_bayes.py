__author__ = 'jiachiliu'

from nulearn.dataset import load_spambase
from nulearn.bayes import BernoulliNaiveBayes
from nulearn.cross_validation import train_test_shuffle_split
from nulearn.validation import *


train, target = load_spambase()

overall_means = [train[:, f].mean() for f in range(train.shape[1])]
overall_std = [train[:, f].std() for f in range(train.shape[1])]

train, test, train_target, test_target = train_test_shuffle_split(train, target, len(train) / 10)

cf = BernoulliNaiveBayes(overall_means)
predicts = cf.fit(train, train_target).predict_class(test)

cm = confusion_matrix(test_target, predicts)
print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
er, acc, fpr, tpr = confusion_matrix_analysis(cm)
print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)