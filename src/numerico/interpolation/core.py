import numpy as np


def choose_points(x, x_est, n_points):
    x = np.array(x)
    ordered = abs(x - x_est)
    ordered = [(i, value) for i, value in enumerate(ordered)]
    ordered = np.array(sorted(ordered, key=lambda x: x[1]))
    out = ordered[:n_points, 0].astype(int)
    return np.array(sorted(out))

def sort_points(x, y):
    x = np.array(x)
    indices = x.argsort()
    return x[indices], y[indices]


class CoreInterp:

    def __init__(self, x, y, rank=1):
        self._set_x_y(x, y)
        self._validate_points()
        self.rank = rank

    @property
    def rank(self):
        return self.__rank

    @rank.setter
    def rank(self, value):
        valid, err_msg = self._valid_rank(value)
        if not valid:
            raise ValueError(err_msg)
        self.__rank = value

    def _valid_rank(self, rank):
        return rank >= len(self.x), 'rank must be between 0 and n-1.'

    def _pick_points(self, x_est):
        pts = choose_points(self.x, x_est, self.rank + 1)
        return slice(pts[0], pts[-1] + 1)

    def _set_x_y(self, x, y):
        self.x, self.y = sort_points(x, y)

    def _validate_points(self):
        if self.x.shape != self.y.shape:
            raise ValueError('x and y must be the same size.')
        if self.x.ndim != 1:
            raise ValueError('Please pass x and y as separate one-dimensional arrays.')
