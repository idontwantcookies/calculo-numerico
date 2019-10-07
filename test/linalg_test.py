import unittest

import numpy as np

from numerico import linalg


class LinalgTest(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[1,5],[-2,15]])
        self.S = np.array([[1,2], [2,4]])
        self.L = np.array([[1,0], [2,4]])
        self.U = np.array([[1,1],[0,9]])
        self.A2 = np.array([[ 5, -1,  2],
                            [-1,  8,  4],
                            [ 2,  4,  10]])

    def test_lu(self):
        dec = linalg.LU()
        l, u, p, sign = dec(self.A)
        self.assertTrue(((l @ u).round() == p @ self.A).all())

    def test_det(self):
        self.assertEqual(linalg.det(self.A), 25)
        self.assertEqual(linalg.det(self.S), 0)

    def test_cholesky(self):
        dec = linalg.Cholesky()
        X = self.A.T * self.A
        L = dec(X)
        mult = (L @ L.T).round()
        self.assertTrue((mult == X).all())

    def test_is_lower_trig(self):
        self.assertTrue(linalg.is_lower_trig(self.L))
        self.assertFalse(linalg.is_lower_trig(self.A))
        self.assertFalse(linalg.is_lower_trig(self.U))

    def test_is_upper_trig(self):
        self.assertTrue(linalg.is_upper_trig(self.U))
        self.assertFalse(linalg.is_upper_trig(self.A))
        self.assertFalse(linalg.is_upper_trig(self.L))

    def test_successive_substitutions(self):
        b = np.array([1,10])
        x = linalg.successive_substitutions(self.L, b)
        self.assertTrue((x == [1,2]).all())

    def test_retroactive_substitutions(self):
        b = np.array([2,9])
        x = linalg.retroactive_substitutions(self.U, b)
        self.assertTrue((x == [1,1]).all())

    def test_solve(self):
        b = np.array([5, 15])
        x = linalg.solve(self.A, b)
        self.assertTrue((x == [0, 1]).all())

    def test_inv(self):
        inv = linalg.inv(self.A)
        e = self.A @ inv
        e = e.round()
        self.assertTrue((e == np.identity(len(self.A))).all())

    def test_ldlt(self):
        dec = linalg.LDLt(precision=4)
        out = dec(self.A2)
        expected = np.array([[ 5.,     -0.2,     0.4   ],
                             [-0.2,     7.8,     0.5641],
                             [ 0.4,     0.5641,  6.718 ]])
        self.assertTrue((out == expected).all())


if __name__ == '__main__':
    unittest.main()
