__author__ = 'jiachiliu'

import numpy as np
from nulearn.neural_network import NeuralNetwork
from nulearn.preprocessing import append_new_column


def main():
    layers = [8, 3, 8]
    train = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]])

    target = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1]])
    nn = NeuralNetwork(layers)
    print 'weights: ', nn.weights
    print 'bias: ', nn.bias

    nn.fit(train, target, 0.3, 20000)
    print '\n==============final weights============='
    print 'Layer 1: \n', nn.weights[0]
    print 'Layer 2: \n', nn.weights[1]

    print '============== Predict ============='
    for t in train:
        result = nn.predict(t, 0.8)
        print "%s -> %s -> %s" % (result[0], result[1], result[3])


if __name__ == '__main__':
    main()