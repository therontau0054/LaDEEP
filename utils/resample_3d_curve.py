import numpy as np
from scipy.interpolate import interp1d

def resample_3d_curve(curve, num_points):
    t = np.linspace(0, 1, len(curve))
    t_new = np.linspace(0, 1, num_points)
    
    resampled_curve = np.zeros((num_points, 3))
    for i in range(3):
        f = interp1d(t, curve[:,i], kind = 'cubic')
        resampled_curve[:,i] = f(t_new)
    
    return resampled_curve