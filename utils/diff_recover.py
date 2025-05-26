import numpy as np

def diff_recover(diff):
    """
    input size: n * 3
    """
    diff = np.concatenate((np.zeros((1, 3), dtype = float), diff), axis = 0)
    return np.cumsum(diff, axis = 0)
