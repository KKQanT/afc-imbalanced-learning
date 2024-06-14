from scipy.spatial.distance import cdist
import numpy as np

def laplacian_kernel(XA, XB, gramma=1.0):
  l1_norm = cdist(XA, XB, 'minkowski', p=1)
  return np.exp(-gramma * l1_norm)