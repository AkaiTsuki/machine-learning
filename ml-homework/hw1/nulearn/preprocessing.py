__author__ = 'jiachiliu'

import numpy as np


def normalize(data, columns=None):
    if columns is None:
        ZeroMeanUnitVariation(range(data.shape[1])).scale(data)
    else:
        ZeroMeanUnitVariation(columns).scale(data)


def append_new_column(data, values, pos):
        return np.insert(data, pos, values=values, axis=1)

class ZeroMeanUnitVariation:
    def __init__(self, cols):
        self.cols = cols
        self.meta = {}

    def scale(self, dataset):
        for col in self.cols:
            col_values = dataset[:, col]
            mean = col_values.mean()
            std = col_values.std()
            self.meta[col] = (mean, std)

        for i in range(dataset.shape[0]):
            for c in self.cols:
                dataset[i][c] = (dataset[i][c] - self.meta[c][0]) / self.meta[c][1]
