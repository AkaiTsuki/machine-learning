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
    # sigma = [np.cov(data, rowvar=0)] * K
    sigma = [np.identity(n)] * K
    gamma = np.zeros((K, m))
    count = 0

    # pi = np.array([0.3206959273694992, 0.45357304148330913, 0.22573103114719137])
    # mu = np.array([[6.95145515, 4.2786734], [4.93326615, 7.06691818], [3.27472168, 3.19007281]])
    # sigma = [
    #     np.array([[1.05843789, 0.20623345],  # covariance 1
    #               [0.20623345, 1.33288348]]),
    #     np.array([[0.97443921, 0.25982045],  # covariance 2
    #               [0.25982045, 0.94619833]]),
    #     np.array([[1.44474827, 0.33745813],  # covariance 3
    #               [0.33745813, 3.69796638]])
    # ]

    while count < max_iter:
        # E step
        print '==============iteration %s =================' % (count + 1)
        for p in range(m):
            gaussian_vector = gaussians(data[p], n, mu, sigma, pi)
            s = sum(gaussian_vector)
            for k in range(K):
                gamma[k][p] = gaussian_vector[k] / s
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

        print 'mean : %s' % mu.tolist()
        print 'sigma : %s' % sigma
        print 'pi : %s' % pi.tolist()
        print ' '
        count += 1
    return gamma


def gaussians(x, n, mu, sigma, pi):
    s = []
    for k in range(len(mu)):
        s.append(pi[k] * gaussian(x, n, mu[k], sigma[k]))
    return s


def gaussian(x, n, mu, sigma):
    v = 1. / (((2 * np.pi) ** (n / 2)) * (det(sigma) ** 0.5))
    diff = x - mu
    v *= np.exp(-0.5 * diff.T.dot(inv(sigma)).dot(diff))
    return v


def plot(gamma, data):
    m = len(data)
    data_splits = {}
    gamma_splits = {}

    for p in range(m):
        inx = np.argmax(gamma[:, p])
        if inx not in data_splits:
            data_splits[inx] = []
            gamma_splits[inx] = []
        data_splits[inx].append(data[p])
        gamma_splits[inx].append(gamma[inx])

    colors = ['r', 'g', 'b']

    for k, sub_data in data_splits.items():
        print 'n%s = %s' % (k, len(sub_data))
        split = np.array(sub_data)
        plt.scatter(split[:, 0], split[:, 1], marker='o', c=colors[k])
    plt.show()


def gaussian_3():
    gaussian3 = load_3gaussian()
    plot(em(gaussian3, 3), gaussian3)


def gaussian_2():
    gaussian2 = load_2gaussian()
    plot(em(gaussian2, 2), gaussian2)

if __name__ == '__main__':
    gaussian_3()
