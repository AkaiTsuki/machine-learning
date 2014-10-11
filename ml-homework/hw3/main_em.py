__author__ = 'jiachiliu'

from nulearn.dataset import load_2gaussian, load_3gaussian
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det, inv
from mpl_toolkits.mplot3d import Axes3D


def em(data, K, max_iter=40):
    m, n = data.shape
    pi = np.array([1.0 / K] * K)
    mu = np.random.rand(K, n)
    sigma = [np.cov(data, rowvar=0)] * K
    gamma = np.zeros((K, m))
    count = 0
    while count < max_iter:
        # E step
        print '==============iteration %s =================' % (count + 1)
        for k in range(K):
            for p in range(m):
                gamma[k][p] = pi[k] * gaussian(data[p], n, mu[k], sigma[k]) / gaussians(data[p], n, mu, sigma, pi)

        print 'gamma : %s' % gamma

        # M step
        for k in range(K):
            sum_gamma_k = gamma[k].sum()
            # update mu
            mu[k] = np.zeros(n)
            for p in range(m):
                mu[k] += gamma[k][p] * data[p] / sum_gamma_k
            # update sigma
            sigma[k] = np.zeros(sigma[k].shape)
            for p in range(m):
                diff = np.atleast_2d(data[p] - mu[k])
                sigma[k] += (gamma[k][p] * diff.T.dot(diff)) / sum_gamma_k
            # update pi
            pi[k] = sum_gamma_k / m

        print 'mean : %s' % mu
        print 'sigma : %s' % sigma
        print 'pi : %s' % pi
        print ' '
        count += 1


    n1_data = []
    n2_data = []
    n1_gamma = []
    n2_gamma = []
    for p in range(m):
        if gamma[0][p] > gamma[1][p]:
            n1_data.append(data[p])
            n1_gamma.append(gamma[0][p])
        else:
            n2_data.append(data[p])
            n2_gamma.append(gamma[1][p])
    print 'n1=%s, n2=%s' % (len(n1_data), len(n2_data))
    n1_data = np.array(n1_data)
    n2_data = np.array(n2_data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(n1_data[:, 0], n1_data[:, 1], n1_gamma, marker='o', c='g')
    ax.scatter(n2_data[:, 0], n2_data[:, 1], n2_gamma, marker='o', c='b')
    # plt.scatter(n1_data[:, 0], n1_data[:, 1], marker='o', c='g')
    # plt.scatter(n2_data[:, 0], n2_data[:, 1], marker='o', c='b')
    plt.show()


def gaussians(x, n, mu, sigma, pi):
    s = 0
    for k in range(len(mu)):
        s += pi[k] * gaussian(x, n, mu[k], sigma[k])
    return s


def gaussian(x, n, mu, sigma):
    v = 1. / (((2 * np.pi) ** (n / 2)) * (det(sigma) ** 0.5))
    diff = x - mu
    v *= np.exp(-0.5 * diff.T.dot(inv(sigma)).dot(diff))
    return v


if __name__ == '__main__':
    gaussian2 = load_2gaussian()
    # gaussian3 = load_3gaussian()

    # plt.scatter(gaussian2[:, 0], gaussian2[:, 1])
    # plt.show()

    em(gaussian2, 2)