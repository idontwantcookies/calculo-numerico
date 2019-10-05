import numpy as np

from .lagrange import Lagrange


class Newton(Lagrange):
    def setUp(self):
        self.n = len(self.x)
        self.build_dely()

    def build_dely(self):
        self.dely = np.zeros((self.n, self.n))
        self.dely[:, 0] = self.y
        for j in range(1, self.n):
            for i in range(self.n - j):
                self.dely[i, j] = (self.dely[i+1, j-1] - self.dely[i, j-1]) / (self.x[i+j] - self.x[i])

    def __call__(self, x_est):
        out = 0
        for i in range(self.n):
            prod = self.dely[0, i]
            for j in range(i):
                prod *= x_est - self.x[j]
            out += prod
        return out
