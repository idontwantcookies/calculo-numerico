import unittest

from numerico import eqroot
from numerico.eqroot.core import briot_ruffini


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
        p = []
