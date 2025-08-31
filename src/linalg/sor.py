import numpy as np

from src.linalg.gauss_seidel import GaussSeidel


class SOR(GaussSeidel):
    # metodo da sobre-relaxação sucessiva (successive over-relaxation)

    def __init__(self, a: np.ndarray, omega, *args, **kwargs):
        self.omega = omega
        super().__init__(a, *args, **kwargs)

    def _build_M(self):
        self.M = self.a.copy().astype(float)
        for i in range(self.n):
            self.M[i] /= -self.a[i, i]
            self.M[i] *= self.omega
            self.M[i, i] = 0

    def _calc_xi(self, i):
        self.x[i] = self.M[i] @ self.x + self.b[i]
        self.x[i] += (1 - self.omega) * self.x[i]
