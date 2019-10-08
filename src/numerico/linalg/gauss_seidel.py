import numpy as np

from .jacobi import Jacobi

class GaussSeidel(Jacobi):
    def __call__(self, b, x0=None):
        self.err = None
        self.b = b
        self._build_b()
        if x0 is None: x0 = np.zeros_like(b).astype(float)
        self.x = x0
        self.iter = 0
        while self.iter < self.max_iter:
            old_x = self.x.copy()
            self._debug()
            for i in range(self.n):
                self.x[i] = self.M[i] @ self.x + self.b[i]
            corr = self.x - old_x
            self.err = max(abs(corr)) / max(abs(self.x))
            if self.err < self.max_err: break
            self.iter += 1
        return self.x