import numpy as np


def unit_vector(vector):
    divider = np.linalg.norm(vector)
    return (vector / divider) if divider != 0 else vector


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
