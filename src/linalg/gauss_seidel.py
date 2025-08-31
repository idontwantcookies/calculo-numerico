import numpy as np

from src.linalg.jacobi import Jacobi


class GaussSeidel(Jacobi):
    def _calc_xi(self, i):
        self.x[i] = self.M[i] @ self.x + self.b[i]

    def solve(self, b, x0=None):
        self.err = None
        self.b = np.array(b)
        self._build_b()
        self.x = self._build_x0() if x0 is None else x0.copy()
        self.iter = 0
        while self.iter < self.max_iter:
            old_x = self.x.copy()
            self._debug()
            for i in range(self.n):
                self._calc_xi(i)
            corr = self.x - old_x
            self.err = max(abs(corr)) / max(abs(self.x))
            if self.err < self.max_err:
                break
            self.iter += 1
        return self.x
