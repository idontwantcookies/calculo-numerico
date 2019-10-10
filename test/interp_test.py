import unittest

import numpy as np

from numerico import interp


def random_points(n=10):
    while True:
        x = np.random.randint(1, 100, size=(n))
        if len(set(x)) == len(x): break
    y = np.random.randint(1, 100, size=(n))
    return x, y


class InterpTest(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0.1, 0.6, 0.8])
        self.y = np.array([1.221, 3.320, 4.953])

    def test_vandermonde_ranks(self):
        x, y = random_points()
        for i in range(1, 6):
            vdm = interp.Vandermonde(x, y, rank=i)
            for x_i, y_i in zip(x, y):
                self.assertEqual(round(vdm(x_i)), y_i)

    def test_lagrange1(self):
        lg = interp.Lagrange(self.x, self.y, rank=1)
        pred = lg(x_est=0.2)
        pred = round(pred, 4)
        self.assertEqual(pred, 1.6408)

    def test_lagrange2(self):
        lg = interp.Lagrange(self.x, self.y, rank=2)
        pred = lg(x_est=0.2)
        pred = round(pred, 4)
        self.assertEqual(pred, 1.4141)

    def test_newton1(self):
        n = interp.Newton(self.x, self.y, rank=1)
        pred = n(x_est=0.2)
        pred = round(pred, 4)
        self.assertEqual(pred, 1.6408)

    def test_newton2(self):
        n = interp.Newton(self.x, self.y, rank=2)
        pred = n(x_est=0.2)
        pred = round(pred, 4)
        self.assertEqual(pred, 1.4141)

    def test_gregory_newton(self):
        x = np.array([110, 120, 130])
        y = np.array([2.041, 2.079, 2.114])
        gn = interp.GregoryNewton(x, y, rank=2)
        pred = gn(115)
        pred = round(pred, 4)
        self.assertEqual(pred, 2.0604)

if __name__ == '__main__':
    unittest.main()
