import numpy as np

from src.interpolation.core import sort_points

coefs = {
    1: np.array([1, 1]),
    2: np.array([1, 4, 1]),
    3: np.array([1, 3, 3, 1]),
    4: np.array([7, 32, 12, 32, 7]),
    5: np.array([19, 75, 50, 50, 75, 19]),
    6: np.array([41, 216, 27, 272, 27, 216, 41]),
    7: np.array([751, 3577, 1323, 2989, 2989, 1323, 3577, 751]),
    8: np.array([989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989]),
}


class NewtonCotes:
    def __init__(self, x, y, rank=1):
        self.rank = rank
        self._set_x_y(x, y)
        self._validate_points()
        self.coefs = coefs[rank]

    def _set_x_y(self, x, y):
        self.x = x.copy()
        if hasattr(y, '__call__'):
            f_i = []
            for xi in x:
                f_i.append(y(x))
            self.y = np.array(f_i)
        else:
            self.y = y.copy()
        self.x, self.y = sort_points(self.x, self.y)
        self.n = len(self.x)

    def _x_evenly_spaced(self):
        diff = self.x[1] - self.x[0]
        for i in range(1, self.n - 1):
            if self.x[i+1] - self.x[i] != diff:
                return False
        self.h = diff
        return True

    def _validate_points(self):
        if self.x.shape != self.y.shape or self.x.ndim != 1:
            raise ValueError('x and y must have the same shape and be 1d.')
        elif (len(self.x) - 1) % self.rank != 0 or len(self.x) < self.rank + 1:
            raise ValueError('Size of x minus 1 must be divisible by rank and '
                             'at least equal to rank.')
        elif not self._x_evenly_spaced():
            raise ValueError('x coordinates must be evenly spaced.')

    def __call__(self):
        out = 0
        start = 0
        stop = self.rank + 1
        for i in range((self.n - 1) // self.rank):
            out += self.y[start:stop] @ self.coefs
            start = stop - 1
            stop += self.rank
        out *= self.rank * self.h / sum(self.coefs)
        return out
