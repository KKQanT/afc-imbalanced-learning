import numpy as np
from scipy.spatial.distance import cdist


def conformal_transform_kernel(X, Y, computed_kernel, support_vectors, tau_squareds):
    # K'(Xi, Xj) = D(Xi)D(Xk)K(Xi, Xj)
    D_X = calculate_D(X, support_vectors, tau_squareds)
    D_Y = calculate_D(Y, support_vectors, tau_squareds)
    D_X = D_X.reshape((-1, 1))
    D_Y = D_Y.reshape((1, -1))
    D_XY = np.matmul(D_X, D_Y)
    return np.multiply(D_XY, computed_kernel)


def calculate_D(X, support_vectors, tau_squared):
    #compute D(X)
    l1_dist = cdist(X, support_vectors, "minkowski", p=1)
    return np.exp(-l1_dist / tau_squared).sum(axis=1)


def calculate_tau_squared(distance_mat, eta=None):
    
    #example: 
    #given we now interest in negative support vectors (have N instances)
    #the other set of support vectors will be positive class (have P instances)
    
    #In this case, distance_mat will be N rows X P cols where row i and col j represent rieman distance
    #from ith negative support vector to jth positive support vector

    #calculate max min rieman distance for each support vectors
    #then M is equal to the average value of the min max distance
    #there fore M will be a vector of an average rieman distance corresponding to support vector ith 
    M = (np.max(distance_mat, axis=1) + np.min(distance_mat, axis=1)) / 2
    #filter distance that less than M as suggested in paper
    mask = distance_mat < M.reshape(-1, 1)
    masked_distance = distance_mat * mask

    if eta is None:

        if distance_mat.shape[0] < distance_mat.shape[1]:
            eta = 1
        else:
            eta = distance_mat.shape[1] / distance_mat.shape[0]
    
    #compute average filtered distance
    # masked_distance.sum(axis=1) is the sum of filtered distance
    # mask.sum(axis=1) is count of filtered distance
    # thus tau_squared is vector of average distance
    tau_squared = masked_distance.sum(axis=1) / mask.sum(axis=1)

    return tau_squared * eta
