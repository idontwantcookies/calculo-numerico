import numpy as np


class Jacobi:
    def __init__(self, a, debug=False, precision=None, max_iter=500,
                 max_err=1e-5):
        self.a = a
        self.n = len(a)
        self._build_M()
        self.debug = debug
        self.max_iter = max_iter
        self.max_err = max_err

    @property
    def converges(self):
        for i in range(self.n):
            if self.a[i,i] <= abs(self.a[i,:i]).sum() + abs(self.a[i,i+1:]).sum():
                return False
        else:
            return True

    def _build_M(self):
        self.M = self.a.copy().astype(float)
        for i in range(self.n):
            self.M[i] /= -self.a[i,i]
            self.M[i,i] = 0

    def _build_b(self):
        self.b = self.b.copy().astype(float)
        for i in range(self.n):
            self.b[i] /= self.a[i,i]

    def _debug(self):
        if self.debug:
            print(self.iter, *self.x, self.err, sep='\t')

    def __call__(self, b, x0=None):
        self.err = None
        self.b = b
        self._build_b()
        if x0 is None: x0 = np.zeros_like(b).astype(float)
        self.x = x0
        self.iter = 0
        while self.iter < self.max_iter:
            self._debug()
            new_x = self.M @ self.x + self.b
            corr = new_x - self.x
            self.x = new_x
            self.err = max(abs(corr)) / max(abs(self.x))
            if self.err < self.max_err: break
            self.iter += 1
        return self.x
