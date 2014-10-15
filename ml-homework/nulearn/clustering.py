__author__ = 'jiachiliu'

import numpy as np


class KMeans:
    def __init__(self):
        self.clusters = {}
        self.means = None

    def fit(self, train, K=2, max_iter=40):
        n, d = train.shape
        self.means = self.init_means(d, K)
        count = 0
        while count < max_iter:
            print count + 1
            # Assignment step
            dists = np.zeros((K, n))
            for i in range(n):
                t = train[i]
                for k in range(K):
                    dists[k][i] = self.distance(t, self.means[k])

            for k in range(K):
                self.clusters[k] = []

            for i in range(n):
                t = train[i]
                c = np.argmin(dists[:, i])
                self.clusters[c].append(t)

            # Update step
            for k in range(K):
                cluster = self.clusters[k]
                m = np.zeros(d)
                for t in cluster:
                    m += t
                self.means[k] = m / len(cluster)
                self.clusters[k] = np.array(self.clusters[k])

            count += 1


    @staticmethod
    def distance(t, m):
        return np.linalg.norm(t - m)

    @staticmethod
    def init_means(n, K):
        means = []
        for k in range(K):
            m = np.random.rand(n)
            means.append(m)
        return np.array(means)
