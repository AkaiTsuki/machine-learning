__author__ = 'jiachiliu'

import numpy as np
import sys


class Unit:
    def __init__(self, weights_count):
        # A list of weights that come in to the unit
        self.weights = np.zeros(weights_count)
        # the bias of unit
        self.bias = 0.0

    @staticmethod
    def weighted_sum(w, x):
        return np.dot(x, w)

    @staticmethod
    def logistic(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def logistic_derivative(x):
        return Unit.logistic(x) * (1 - Unit.logistic(x))

    def __str__(self):
        return '(weights: ' + str(self.weights) + ' bias: ' + str(self.bias) + ')'


class NeuralNetwork:
    def __init__(self, layers):
        # a list contains number of units in each layer
        self.layers = layers
        # the weights of the neural network
        # weights[l][m][n] represents the weight outcomes from layer l to layer l+1
        # from node m to node n
        self.weights = self.init_weights(layers)
        # the bias for each unit
        # bias[i][j] represents the bias of unit j in layer i
        self.bias = self.init_bias(layers)

    def init_weights(self, layers):
        weights = []
        for l in range(1, len(layers)):
            weights.append(2 * np.random.random((layers[l - 1], layers[l])) * 0.25)
        return weights

    def init_bias(self, layers):
        bias = []
        for l in range(1, len(layers)):
            bias.append(2 * np.random.random(layers[l]) * 0.25)
        return bias

    @staticmethod
    def logistic(x):
        return 1.0 / (1 + np.exp(-x))

    def print_progress(self, k, max_loop):
        sys.stdout.write("\rProgress: %s/%s" % (k+1, max_loop))
        sys.stdout.flush()

    def predict(self, test, threshold=0.8):
        predict = test
        result = [predict]
        for l in range(len(self.weights)):
            predict = self.logistic(np.dot(self.weights[l].T, predict) + self.bias[l])
            result.append(predict)
        result.append(np.array(map(lambda v: 1 if v >= threshold else 0, predict)))
        return result

    def fit(self, train, target, rate=0.5, epoches=1000):
        for loop in range(epoches):
            self.print_progress(loop, epoches)
            rand = np.random.randint(train.shape[0])
            t = train[rand]
            y = target[rand]

            # feed forward
            # the output for each unit
            # for input layer, the output is same as input
            # which is the tuple
            outputs = [t]

            # for hidden and output layer, compute the output
            for l in range(len(self.weights)):
                weights = self.weights[l]
                # m is number of units in lower layer
                # n is number of units in higher layer
                m, n = weights.shape
                # the outputs for all unit in layer l
                output = []
                # for each unit in layer l+1
                for u in range(n):
                    bias = self.bias[l][u]
                    # unit input
                    u_i = np.dot(outputs[-1], weights[:, u]) + bias
                    # unit output
                    u_o = self.logistic(u_i)
                    output.append(u_o)
                outputs.append(np.array(output))
            # print 'outputs:', outputs

            # back propagation
            # for output layer
            errors = [outputs[-1] * (1 - outputs[-1]) * (y - outputs[-1])]
            # for each hidden layers from the last hidden layer to first hidden layer
            for l in range(len(outputs) - 2, 0, -1):
                # the errors for each unit in layer l
                error = []

                # the weights of higher layer
                weights = self.weights[l]
                # m is number of units in lower layer
                # n is number of units in higher layer
                m, n = weights.shape
                for u in range(m):
                    e = outputs[l][u] * (1 - outputs[l][u]) * np.dot(errors[-1], weights[u])
                    error.append(e)
                errors.append(np.array(error))
            errors.reverse()
            # print 'errors: ', errors

            # update weights
            for l in range(len(self.weights)):
                for m in range(len(self.weights[l])):
                    for n in range(len(self.weights[l][m])):
                        self.weights[l][m][n] += rate * errors[l][n] * outputs[l][m]

            # update bias
            for l in range(len(self.bias)):
                for m in range(len(self.bias[l])):
                    self.bias[l][m] += rate * errors[l][m]








