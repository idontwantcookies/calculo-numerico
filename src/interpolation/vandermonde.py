import numpy as np

from src.interpolation.core import choose_points, sort_points
from src.linalg.lu import LU


class Vandermonde:
    def __init__(self, x, y, rank=1):
        self.x, self.y = np.array(x), np.array(y)
        self.rank = rank
        self._sort()
        self.validate_points()
        self.setUp()

    @property
    def rank(self):
        return self.__rank

    @rank.setter
    def rank(self, value):
        if value >= len(self.x):
            raise ValueError('rank must be between 0 and n-1.')
        self.__rank = value

    def _sort(self):
        self.x, self.y = sort_points(self.x, self.y)

    def setUp(self):
        n = len(self.x) + 1
        self.V = np.zeros((len(self.x), n))
        self.V[:, 0] = 1
        for i in range(1, n):
            self.V[:, i] = self.x ** i

    def validate_points(self):
        if self.x.shape != self.y.shape:
            raise ValueError('x and y must be the same size.')
        if self.x.ndim != 1:
            raise ValueError(
                'Please pass x and y as separate one-dimensional arrays.')

    def coefs(self, slc):
        lu = LU(self.V[slc, :self.rank + 1])
        out = lu.solve(self.y[slc])
        out = list(reversed(out))
        out = np.array(out)
        out = np.poly1d(out)
        return out

    def _pick_points(self, x_est):
        pts = choose_points(self.x, x_est, self.rank + 1)
        return slice(pts[0], pts[-1] + 1)

    def __call__(self, x_est):
        slc = self._pick_points(x_est)
        poly = self.coefs(slc)
        return poly(x_est)
