__author__ = 'jiachiliu'

from nulearn import dataset
from nulearn import preprocessing
from nulearn.linear_model import Perceptron


def perceptron():
    train, target = dataset.load_perceptron()
    train = preprocessing.append_new_column(train, 1.0, 0)
    classifier = Perceptron()
    classifier.fit(train, target, max_loop=30)

    print classifier.weights
    print classifier.weights / -classifier.weights[0]


def main():
    perceptron()


if __name__ == '__main__':
    main()