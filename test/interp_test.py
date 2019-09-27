import unittest

import numpy as np

from numerico import interp


def random_vandermonde(rank=1):
    while True:
        x = np.random.randint(1, 100, size=(rank+1))
        y = np.random.randint(1, 100, size=(rank+1))
        if len(set(x)) == len(x): break
    poly = interp.vandermonde(x, y)
    return x, y, poly

class InterpTest(unittest.TestCase):
    def test_vandermonde_ranks(self):
        for i in range(1, 6):
            x, y, p = random_vandermonde(rank=i)
            for x_i, y_i in zip(x, y):
                self.assertEqual(round(np.polyval(p, x_i)), y_i)

if __name__ == '__main__':
    unittest.main()
