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
        pred = lg(x_est=0.2, rank=2)
        self.assertEqual(pred, 1.4141142857142863)

if __name__ == '__main__':
    unittest.main()
