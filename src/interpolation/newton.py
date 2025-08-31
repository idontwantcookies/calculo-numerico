import numpy as np

from src.interpolation.lagrange import Lagrange


class Newton(Lagrange):
    def setUp(self):
        self.n = len(self.x)
        self.build_dely()

    def build_dely(self):
        self.dely = np.zeros((self.n, self.n))
        self.dely[:, 0] = self.y
        for j in range(1, self.n):
            for i in range(self.n - j):
                self.dely[i, j] = (self.dely[i+1, j-1] -
                                   self.dely[i, j-1]) / (self.x[i+j] - self.x[i])

    def get_dely(self, i, order):
        if i + order > self.rank:
            raise ValueError(
                f'dely for order {order} and point {i} is not set on rank {self.rank}.')
        return self.dely[i, order]

    def __call__(self, x_est):
        slc = self._pick_points(x_est)
        out = 0
        for order in range(self.rank + 1):
            prod = self.get_dely(slc.start, order)
            for j in range(slc.start, order):
                prod *= x_est - self.x[j]
            out += prod
        return out
