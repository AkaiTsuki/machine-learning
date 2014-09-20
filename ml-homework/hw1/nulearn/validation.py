__author__ = 'jiachiliu'

import math
import logging
import numpy as np


def mse(pred, act):
    mse = 0.0
    E = abs(pred - act)
    for i in range(len(E)):
        mse += E[i] ** 2
    return mse / len(E)


def rmse(pred, act):
    return math.sqrt(mse(pred, act))


def mae(pred, act):
    mae = 0.0
    E = abs(pred - act)
    for i in range(len(E)):
        mae += abs(E[i])
    return mae / len(E)


def confusion_matrix_analysis(cm):
    true_negative = cm[0, 0]
    false_positive = cm[0, 1]
    false_negative = cm[1, 0]
    true_positive = cm[1, 1]

    total = true_negative + true_positive + false_negative + false_positive
    neg = true_negative + false_positive
    pos = false_negative + true_positive
    error_rate = 1.0 * (false_positive + false_negative) / total
    accuracy = 1.0 * (true_negative + true_positive) / total
    if neg == 0:
        fpr = 0
    else:
        fpr = 1.0 * false_positive / neg
    if pos == 0:
        tpr = 0
    else:
        tpr = 1.0 * true_positive / pos

    return error_rate, accuracy, fpr, tpr


def confusion_matrix(actual, predict):
    true_negative = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0

    for i in range(len(actual)):
        act_label = actual[i]
        pred_label = predict[i]
        if act_label == 1 and pred_label == 0:
            false_negative += 1
        elif act_label == 0 and pred_label == 1:
            false_positive += 1
        elif act_label == 1 and pred_label == 1:
            true_positive += 1
        else:
            true_negative += 1

    return np.array([[true_negative, false_positive], [false_negative, true_positive]])