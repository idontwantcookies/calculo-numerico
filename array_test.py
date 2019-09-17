import unittest
import pdb

from array_core import Array


class TestArray(unittest.TestCase):
    def setUp(self):
        self.x = Array([1,2,3])
        self.y = Array([3,2,1])
        self.bit1 = Array([True, False, False])
        self.bit2 = Array([True, True, False])

    def test_eq(self):
        self.assertCountEqual(self.x, [1,2,3])
        self.assertEqual(self.x == [1,2,3], [True, True, True])
        with self.assertRaises(IndexError):
            self.x == [1,2]

    def test_neq(self):
        self.assertTrue(all(self.x != [10,20,30]))
        self.assertCountEqual(self.x != [1,2,3], [False, False, False])
        with self.assertRaises(IndexError):
            self.assertTrue(self.x != [1,2])

    def test_add(self):
        self.assertEqual(self.x + [1,1,1], [2,3,4])
        self.assertEqual([1,1,1] + self.x, [2,3,4])
        self.assertEqual(self.x + 1, [2,3,4])
        self.assertEqual(1 + self.x, [2,3,4])

    def test_sub(self):
        self.assertEqual(self.x - self.x, [0,0,0])
        self.assertEqual(self.x - [1,1,1], [0,1,2])
        self.assertEqual([1,1,1] - self.x , [0,-1,-2])
        self.assertEqual(self.x - 1, [0,1,2])

    def test_mul(self):
        expected = [2,4,6]
        self.assertEqual(self.x * 2, expected)
        self.assertEqual(2 * self.x, expected)
        self.assertEqual(self.x * [2,2,2], expected)
        self.assertEqual([2,2,2] * self.x, expected)

    def test_truediv(self):
        self.assertEqual(self.x / 2, [0.5,1,1.5])
        self.assertEqual(2 / self.x, [2,1,2/3])
        self.assertEqual(self.x / [2,2,2], [0.5,1,1.5])
        self.assertEqual([2,2,2] / self.x, [2,1,2/3])

    def test_floordiv(self):
        self.assertEqual(self.x // 2, [0,1,1])
        self.assertEqual(2 // self.x, [2,1,0])
        self.assertEqual(self.x // [2,2,2], [0,1,1])
        self.assertEqual([2,2,2] // self.x, [2,1,0])

    def test_mod(self):
        self.assertEqual(self.x % 3, [1,2,0])
        self.assertEqual(2 % self.x, [0,0,2])
        self.assertEqual(self.x % [3,3,3], [1,2,0])
        self.assertEqual([2,2,2] % self.x, [0,0,2])

    def test_comparison(self):
        self.assertEqual(self.x > 2, [False, False, True])
        self.assertEqual(self.x > [2,2,2], [False, False, True])
        self.assertEqual(self.x >= 2, [False, True, True])
        self.assertEqual(self.x < 2, [True, False, False])
        self.assertEqual(self.x < [2,2,2], [True, False, False])
        self.assertEqual(self.x <= 2, [True, True, False])
        self.assertEqual(self.x <= [2,2,2], [True, True, False])

    def test_right_comparison(self):
        self.assertEqual(2 < self.x, self.x >= 2)
        self.assertEqual([2,2,2] < self.x, self.x >= [2,2,2])
        self.assertEqual(2 > self.x, self.x <= 2)
        self.assertEqual([2,2,2] > self.x, self.x <= [2,2,2])
        self.assertEqual(2 == self.x, [False, True, False])

    def test_bitwise(self):
        self.assertCountEqual(~self.bit1, [False, True, True])
        self.assertCountEqual(self.bit1 & self.bit1, self.bit1)
        self.assertCountEqual(self.bit1 & self.bit2, [True, False, False])
        self.assertCountEqual(self.bit1 | self.bit2, [True, True, False])
        self.assertCountEqual(self.bit1 ^ self.bit2, [False, True, False])

    def test_matmul(self):
        self.assertEqual(self.x @ self.y, 10)
        with self.assertRaises(IndexError):
            self.x @ [1,2]
        with self.assertRaises(IndexError):
            self.x @ 2

if __name__ == '__main__':
    unittest.main()
