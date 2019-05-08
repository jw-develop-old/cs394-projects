import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as sklrbf

def linear(x1, x2) :
    return np.dot(x1, x2)

def make_poly_kernel(s) :
    return lambda x1, x2 : (np.dot(x1, x2))**s

def rbf(x1, x2) :
    return sklrbf([x1], [x2])[0][0]