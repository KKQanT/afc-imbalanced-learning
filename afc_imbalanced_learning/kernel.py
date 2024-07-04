from scipy.spatial.distance import cdist
import numpy as np
from sklearn.gaussian_process.kernels import Kernel


def laplacian_kernel(XA, XB, gramma=1.0):
    l1_norm = cdist(XA, XB, "minkowski", p=1)
    return np.exp(-gramma * l1_norm)


class RBFKernel(Kernel):

    def __init__(self, gamma=1):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        pairwise_sq_dists = cdist(X, Y, "sqeuclidean")
        K = np.exp(-self.gamma * pairwise_sq_dists)
        return K


class LaplacianKernel(Kernel):

    def __init__(self, gamma=1):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        l1_norm = cdist(X, Y, "minkowski", p=1)
        return np.exp(-self.gamma * l1_norm)
