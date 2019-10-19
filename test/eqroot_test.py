import unittest

import numpy as np

from numerico import eqroot


class EqRootTest(unittest.TestCase):
    def test_limits(self):
        coef = [24,-14,-13,2,1]
        l = eqroot.Limits(coef)
        min_pos, max_pos = l.pos
        min_neg, max_neg = l.neg
        self.assertEqual(min_pos, 0.631578947368421)
        self.assertEqual(max_pos, 4.741657386773941)
        self.assertEqual(min_neg, -14.0)
        self.assertEqual(max_neg, -0.5760434788494824)

    def test_briot_ruffini(self):
        p = np.poly1d([1,-5,6])
        q, r = eqroot.briot_ruffini(p, 2)
        self.assertTrue((q == np.poly1d([1,-3])).all())
        self.assertEqual(r, 0)

    def test_bissection(self):
        poly = np.poly1d([1,0,-1])
        x, err = eqroot.bissection(poly, 0, 3)
        self.assertEqual(round(x), 1)
