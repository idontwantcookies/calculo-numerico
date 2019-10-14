import pdb
import unittest

import numpy as np

from numerico import integral


class IntegralTest(unittest.TestCase):
    def test_trapezoid_rule(self):
        x = np.array([1, 1.5, 2, 2.5, 3])
        y = np.array([0, 1.3684, 5.5452, 14.3170, 29.6625])
        nc = integral.NewtonCotes(x, y, rank=1)
        area = nc()
        self.assertEqual(area, 18.030925)

    def test_1_3_simpson_rule(self):
        x = np.array([0,0.25,0.5,0.75,1])
        y = np.array([1,0.9412,0.8000,0.6400,0.5000])
        nc = integral.NewtonCotes(x, y, rank=2)
        area = round(nc(), 4)
        self.assertEqual(area, 0.7854)

    def test_3_8_simpson_rule(self):
        x = np.array([1,1.5,2,2.5,3,3.5,4])
        y = np.array([1.0744,1.7433,2.3884,2.9578,3.4529,3.8860,4.2691])
        nc = integral.NewtonCotes(x, y, rank=3)
        area = round(nc(), 4)
        self.assertEqual(area, 8.5633)
