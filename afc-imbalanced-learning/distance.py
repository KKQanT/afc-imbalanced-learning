import numpy as np


def hyperspace_l2_distance_squared(X, Y, kernel):
    K_X = kernel(X, X)
    K_Y = kernel(Y, Y)
    K_XY = kernel(X, Y)
    K_X = np.diag(K_X)
    K_Y = np.diag(K_Y)
    K_X = np.tile(K_X, (K_XY.shape[1], 1)).T
    K_Y = np.tile(K_Y, (K_XY.shape[0], 1))
    return K_X + K_Y - 2 * K_XY
