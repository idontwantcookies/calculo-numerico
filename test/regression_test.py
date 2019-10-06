import unittest

import numpy as np

from numerico import regression


class RegressionTest(unittest.TestCase):
    def setUp(self):
        self.x1 = np.array([[0.3, 2.7, 4.5, 5.9, 7.8]]).T
        self.y = np.array([1.8, 1.9, 3.1, 3.9, 3.3])

    def test_rank1(self):
        reg = regression.Linear(self.x1, self.y)
        pred = round(reg(0.5), 4)
        self.assertEqual(pred, 1.7909)
