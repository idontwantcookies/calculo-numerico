from math import factorial

import numpy as np

from .newton import Newton


class GregoryNewton(Newton):

    def validate_points(self):
        super().validate_points()
        diff = self.x[1] - self.x[0]
        for i in range(1, len(self.x) - 1):
            if self.x[i+1] - self.x[i] != diff:
                raise ValueError('All x coordinates must be equally spaced. '
                    'Use Newton interp for uneven spaced x-coordinates.')
        self.diff = diff

    def build_dely(self):
        self.dely = np.zeros((self.n, self.n))
        self.dely[:, 0] = self.y
        for j in range(1, self.n):
            for i in range(self.n - j):
                self.dely[i, j] = self.dely[i+1, j-1] - self.dely[i, j-1]

    def __call__(self, x_est):
        aux = (x_est - self.x[0]) / self.diff
        out = 0
        for i in range(self.n):
            prod = self.dely[0, i] / factorial(i)
            for j in range(i):
                prod *= aux - j
            out += prod
        return out
