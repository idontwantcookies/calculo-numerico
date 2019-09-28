import numpy as np


class Lagrange:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.validate_points()
        self.setUp()

    def setUp(self):
        self.n = len(self.x)
        self.G = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                self.G[i, j] = self.x[i] - self.x[j]

    def validate_points(self):
        if self.x.shape != self.y.shape:
            raise ValueError('x and y must be the same size.')
        if self.x.ndim != 1:
            raise ValueError('Please pass x and y as separate one-dimensional arrays.')

    def choose_points(self, x_est, n_points):
        ordered = np.array(sorted(self.x))
        ordered = abs(ordered - x_est)
        ordered = [(i, value) for i, value in enumerate(ordered)]
        ordered = np.array(sorted(ordered, key=lambda x: x[1]))
        out = ordered[:n_points, 0].astype(int)
        return out

    def set_diag(self, x_est):
        for i in range(self.n):
            self.G[i,i] = x_est - self.x[i]

    def __call__(self, x_est, rank):
        n_points = rank + 1
        if n_points > self.n:
            raise ValueError(f'You need {n_points} to interpolate rank {rank}, '
                             f'but you only passed {self.n} points.')
        points = self.choose_points(x_est, n_points)
        self.set_diag(x_est)
        out = 0
        for i in points:
            out += self.y[i] / self.G[i].prod()
        out *= self.G.diagonal().prod()
        return out
