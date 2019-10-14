import unittest

import numpy as np

from numerico import interpolation as interp


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

    def test_natural_spline(self):
        x = np.array([1,2,4,6,7])
        y = np.array([2,4,1,3,3])
        spl = interp.NaturalSpline(x, y)
        for i, value in enumerate(x):
            self.assertEqual(round(spl(value)), y[i])

    def test_natural_spline2(self):
        spl = interp.NaturalSpline(self.x, self.y)
        for x, y in zip(self.x, self.y):
            self.assertEqual(round(spl(x), 4), y)

    def test_extrapolated_spline(self):
        x = np.array([1,2,4,6,7])
        y = np.array([2,4,1,3,3])
        spl = interp.NotAKnotSpline(x, y)
        for i, value in enumerate(x):
            self.assertEqual(round(spl(value)), y[i])

    def test_truncation_error(self):
        func = lambda x: 48 * x
        x = np.array([0, 0.2, 0.4])
        y = np.array([1, 1.1232, 1.5312])
        newton = interp.Newton(x, y, rank=2)
        err = newton.trunc_error(0.1, func)
        err = round(err, 4)
        self.assertEqual(err, 0.0096)


if __name__ == '__main__':
    unittest.main()
