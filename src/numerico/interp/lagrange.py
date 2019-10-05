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
        out = 0
        for i in range(self.n):
            out += self.y[i] / self.G[i].prod()
        out *= self.G.diagonal().prod()
        return out
