import unittest
import pdb

from array_core import Array, dim, shape, all_elements_have_same_length


class TestArray(unittest.TestCase):
    def setUp(self):
        self.x = Array([1,2,3])
        self.y = Array([3,2,1])
        self.bit1 = Array([True, False, False])
        self.bit2 = Array([True, True, False])
        self.A = Array([
            [ 1, 2],
            [-1, 1]
        ])

    def test_init(self):
        with self.assertRaises(IndexError):
            Array([[1,2], [2]])

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
            print(self.x @ 2)
            self.x @ 2
        self.assertTrue((self.A @ self.A).full_equal([[-1, 4], [-2, -1]]))
        self.assertTrue((self.A @ self.A[0]).full_equal([5, 1]))

    def test_dimension(self):
        self.assertEqual(dim([1]), 1)
        self.assertEqual(dim(1), 0)
        self.assertEqual(dim('olar'), 0)
        self.assertEqual(dim([1,2,3]), 1)
        self.assertEqual(dim([[1,2,3]]), 2)

    def test_all_elements_have_same_length(self):
        func = all_elements_have_same_length
        self.assertTrue(func(1))
        self.assertTrue(func('olar'))
        self.assertTrue(func([1,2,3,4]))
        self.assertTrue(func(self.A))
        self.assertFalse(func([[1,2,3],[1,2]]))

    def test_shape(self):
        self.assertEqual(shape([1,2,3]), (3,))
        self.assertEqual(shape([[1,2,3]]), (1,3))
        self.assertEqual(shape([[[1,2,3]], [[0,0,1]]]), (2,1,3))

    def test_full_equal(self):
        self.assertTrue(self.x.full_equal([1,2,3]))
        self.assertTrue(self.A.full_equal([[1,2],[-1,1]]))
        self.assertFalse(self.A.full_equal([[1,2],[-1,3]]))

    def test_matrix(self):
        self.assertTrue((1 + self.A).full_equal([[2,3],[0,2]]))
        self.assertTrue((1 - self.A).full_equal([[0,-1],[2,0]]))
        self.assertTrue((2 * self.A).full_equal([[2,4],[-2,2]]))
        self.assertTrue((2 / self.A).full_equal([[2,1],[-2,2]]))

    def test_transpose(self):
        self.assertTrue(self.A.full_equal(self.A.transpose().transpose()))
        m = Array.randint(10,15)
        self.assertTrue(m.full_equal(m.transpose().transpose()))

    def test_randint(self):
        m = Array.randint(2,3,4)
        self.assertEqual(m.shape, (2,3,4))

    def test_zeros(self):
        z = Array.zeros(2, 4)
        self.assertTrue(z.full_equal([[0] * 4] * 2))

if __name__ == '__main__':
    unittest.main()
