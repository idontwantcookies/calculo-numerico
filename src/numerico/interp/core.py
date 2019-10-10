import numpy as np


def choose_points(x, x_est, n_points):
    x = np.array(x)
    ordered = abs(x - x_est)
    ordered = [(i, value) for i, value in enumerate(ordered)]
    ordered = np.array(sorted(ordered, key=lambda x: x[1]))
    out = ordered[:n_points, 0].astype(int)
    return np.array(sorted(out))

def sort_points(x, y):
    indices = x.argsort()
    return x[indices], y[indices]
