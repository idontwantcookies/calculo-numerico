import pdb
from math import factorial

import numpy as np

from .vandermonde import Vandermonde


class Lagrange(Vandermonde):

    def setUp(self):
        self.n = len(self.x)
        self.G = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i):
                self.G[i, j] = self.x[i] - self.x[j]
                self.G[j, i] = -self.G[i, j]

    def set_diag(self, x_est):
        for i in range(self.n):
            self.G[i,i] = x_est - self.x[i]

    def __call__(self, x_est):
        self.set_diag(x_est)
        slc = self._pick_points(x_est)
        out = 0
        for i in range(slc.start, slc.stop):
            out += self.y[i] / self.G[i, slc].prod()
        out *= self.G[slc, slc].diagonal().prod()
        return out

    def trunc_error(self, x_est, dn_func):
        # dn_func é a n+1-ésima derivada da função sendo interpolada
        f_epslon = 0
        prod = 1
        for x in self.x[self._pick_points(x_est)]:
            u = dn_func(x)
            if abs(u) > f_epslon: f_epslon = u
            prod *= x_est - x
        return abs(f_epslon / factorial(self.rank + 1) * prod)
