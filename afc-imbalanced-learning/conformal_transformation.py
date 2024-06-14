import numpy as np
from scipy.spatial.distance import cdist


def conformal_transform_kernel(X, Y, computed_kernel, support_vectors, tau_squareds):
    D_X = calculate_D(X, support_vectors, tau_squareds)
    D_Y = calculate_D(Y, support_vectors, tau_squareds)
    D_X = D_X.reshape((-1, 1))
    D_Y = D_Y.reshape((1, -1))
    D_XY = np.matmul(D_X, D_Y)
    return np.multiply(D_XY, computed_kernel)


def calculate_D(X, support_vectors, tau_squared):
    l1_dist = cdist(X, support_vectors, "minkowski", p=1)
    return np.exp(-l1_dist / tau_squared).sum(axis=1)


def calculate_tau_squared(distance_mat):
    M = (np.max(distance_mat, axis=1) + np.min(distance_mat, axis=1)) / 2
    mask = distance_mat < M.reshape(-1, 1)
    masked_distance = distance_mat * mask

    if distance_mat.shape[0] < distance_mat.shape[1]:
        eta = 1
    else:
        eta = distance_mat.shape[1] / distance_mat.shape[0]
    tau_squared = masked_distance.sum(axis=1) / mask.sum(axis=1)

    return tau_squared * eta
