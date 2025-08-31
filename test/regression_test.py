import unittest

import numpy as np

from src.regression.linear import Linear
from src.regression.polynomial import Polynomial


class RegressionTest(unittest.TestCase):
    def setUp(self):
        self.x1 = np.array([[0.3, 2.7, 4.5, 5.9, 7.8]]).T
        self.x2 = np.array([[0, 1, 2, 3, 4],
                            [-2, -1, 0, 1, 2]]).T
        self.x3 = np.array([[0, 1, 2], [-3, 0, 2]]).T
        self.y2 = np.array([4.1, 3.1, 3.3])
        self.y = np.array([1.8, 1.9, 3.1, 3.9, 3.3])

    def test_simple_regression(self):
        reg = Linear(self.x1, self.y)
        pred = round(reg(0.5), 4)
        self.assertEqual(pred, 1.7909)

    def test_multiple_regression(self):
        reg = Linear(self.x3, self.y2)
        pred = round(reg([1, 1]), 4)
        self.assertEqual(pred, 1.9)

    def test_poly_regression_rank1(self):
        reg = Polynomial(self.x1[:, 0], self.y, rank=1)
        pred = round(reg(0.5), 4)
        self.assertEqual(pred, 1.7909)

    def test_poly_regression_rank2(self):
        reg = Polynomial(self.x1[:, 0], self.y, rank=2)
        pred = round(reg(0.5), 4)
        self.assertEqual(pred, 1.6635)

    def test_poly_regression_interp(self):
        ''' testa se a regressão com 3 pontos produz predições idênticas a y_i,
        ou seja, se a parábola dos mínimos quadrados passa pelos 3 pontos 
        dados.'''
        x = self.x1[:3, 0]
        y = self.y[:3]
        reg = Polynomial(x, y, rank=2)
        for i in range(3):
            pred = reg(x[i])
            pred = round(pred, 1)
            self.assertEqual(pred, y[i])

    def test_poly_regression_quality(self):
        reg = Polynomial(self.x1[:, 0], self.y, rank=1)
        self.assertEqual(round(reg.D, 4), 0.9289)
        self.assertEqual(round(reg.r, 4), 0.8506)
        self.assertEqual(round(reg.r**2, 4), 0.7235)

    def test_linear_regression_quality(self):
        reg = Linear(self.x1, self.y)
        self.assertEqual(round(reg.D, 4), 0.9289)
        self.assertEqual(round(reg.r, 4), 0.8506)
        self.assertEqual(round(reg.r**2, 4), 0.7235)
