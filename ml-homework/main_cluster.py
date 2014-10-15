__author__ = 'jiachiliu'

from nulearn.dataset import load_2gaussian, load_3gaussian
from nulearn.clustering import KMeans
import matplotlib.pyplot as plt


def plot(clusters):
    colors = ['r', 'g', 'b']
    for k in clusters:
        split = clusters[k]
        plt.scatter(split[:, 0], split[:, 1], marker='o', c=colors[k])
    plt.show()


if __name__ == '__main__':
    data = load_3gaussian()
    cf = KMeans()
    K = 3
    cf.fit(data, K=K, max_iter=20)
    for k in range(K):
        print 'n%s = %s' % (k, len(cf.clusters[k]))
    plot(cf.clusters)