import numpy as np


class Lagrange:
    def __init__(self, x, y):
        self.x, self.y = np.array(x), np.array(y)
        self.validate_points()
        self.setUp()

    def setUp(self):
        self.n = len(self.x)
        self.G = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i):
                self.G[i, j] = self.x[i] - self.x[j]
                self.G[j, i] = -self.G[i, j]

    def validate_points(self):
        if self.x.shape != self.y.shape:
            raise ValueError('x and y must be the same size.')
        if self.x.ndim != 1:
            raise ValueError('Please pass x and y as separate one-dimensional arrays.')

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
