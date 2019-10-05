import unittest

import numpy as np

from numerico import interp


def random_points(rank=1):
    while True:
        x = np.random.randint(1, 100, size=(rank+1))
        y = np.random.randint(1, 100, size=(rank+1))
        if len(set(x)) == len(x): break
    return x, y


class InterpTest(unittest.TestCase):
    def test_vandermonde_ranks(self):
        for i in range(1, 6):
            x, y = random_points(rank=i)
            p = interp.vandermonde(x, y)
            for x_i, y_i in zip(x, y):
                self.assertEqual(round(np.polyval(p, x_i)), y_i)

    def test_lagrange(self):
        x = np.array([0.1, 0.6, 0.8])
        y = np.array([1.221, 3.320, 4.953])
        lg = interp.Lagrange(x, y)
        pred = lg(x_est=0.2)
        self.assertEqual(pred, 1.4141142857142863)
        # TODO: not working
        # lg = interp.Lagrange(x[:-1], y[:-1])
        # pred = lg(x_est=0.2)
        # print(lg.G)
        # self.assertEqual(pred, 2.829257142857143)

    def test_newton(self):
        x = np.array([0.1, 0.6])
        y = np.array([1.221, 3.320])
        n = interp.Newton(x, y)
        pred = n(x_est=0.2)
        self.assertEqual(pred, 1.6408)


if __name__ == '__main__':
    unittest.main()
