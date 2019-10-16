import numpy as np

from ..linalg import gauss
from .core import CoreInterp


class NaturalSpline(CoreInterp):
    def __init__(self, x, y):
        super().__init__(x, y, rank=3)
        self._setUp()
        self._solve_system()
        self._calculate_coeficients()

    def _valid_rank(self, rank):
        return True, ''

    def _setUp(self):
        self.n = len(self.x)
        self.dely = np.zeros(self.n - 1)
        self.h = np.zeros(self.n - 1)
        for i in range(self.n - 1):
            self.h[i] = self.x[i+1] - self.x[i]
            self.dely[i] = (self.y[i+1] - self.y[i]) / self.h[i]
        self.s2 = np.zeros(self.n)

    def _set_arbitrary_s2(self):
        self.s2[0] = 0
        self.s2[-1] = 0

    def _build_matrix(self):
        m = np.zeros((self.n - 2, self.n))
        for i in range(self.n - 2):
            m[i, i] = self.h[i]
            m[i, i+1] = 2 * (self.h[i] + self.h[i+1])
            m[i, i+2] = self.h[i+1]
        return m

    def _build_b(self):
        b = np.zeros(self.n - 2)
        for i in range(self.n - 2):
            b[i] = self.dely[i+1] - self.dely[i]
        return b

    def _solve_system(self):
        m = self._build_matrix()
        b = self._build_b()
        self.s2[1:-1], det = gauss(m[:,1:-1], 6 * b, pivoting=False)
        self._set_arbitrary_s2()

    def _calculate_coeficients(self):
        self.splines = []
        for i in range(self.n - 1):
            a = (self.s2[i+1] - self.s2[i]) / (6 * self.h[i])
            b = self.s2[i] / 2
            c = self.dely[i] - (self.s2[i+1] + 2 * self.s2[i]) * self.h[i] / 6
            d = self.y[i]
            self.splines.append(np.poly1d([a, b, c, d]))

    def __call__(self, x_est):
        self.rank = 1   # forçando a escolher 2 pontos
        i = self._pick_points(x_est).start
        self.rank = 3
        return self.splines[i](x_est - self.x[i])
