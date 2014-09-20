__author__ = 'jiachiliu'

import numpy as np


class CsvFileReader:
    """
    CsvFileReader will read data from csv file
    """

    def __init__(self, path):
        self.path = path

    def read(self, delimiter, converter):
        f = open(self.path)
        lines = f.readlines()
        return self.parse_lines(lines, delimiter, converter)

    @staticmethod
    def parse_lines(lines, delimiter, converter):
        data = []
        for line in lines:
            if line.strip():
                row = [s.strip() for s in line.strip().split(delimiter) if s.strip()]
                data.append(row)
        return np.array(data, converter)


def load_spambase():
    reader = CsvFileReader('data/spambase.data')
    data = reader.read(',', float)
    total_col = data.shape[1]
    return data[:, :total_col - 1], data[:, total_col - 1]


def load_boston_house():
    reader = CsvFileReader('data/housing_train.txt')
    train_data = reader.read(' ', float)
    train = train_data[:, :train_data.shape[1] - 1]
    train_target = train_data[:, train_data.shape[1] - 1]

    test_data = CsvFileReader('data/housing_test.txt').read(' ', float)
    test = test_data[:, :test_data.shape[1] - 1]
    test_target = test_data[:, test_data.shape[1] - 1]

    return train, train_target, test, test_target