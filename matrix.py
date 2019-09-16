import random

from vector import Vector
import exceptions


class Matrix:
    @staticmethod
    def identity(size):
        I = Matrix()
        for i in range(size):
            row = Vector()
            for j in range(size):
                row.append(int(i == j))
            I.append(row)
        return I

    @staticmethod
    def zeros(nrows, ncols):
        m = Matrix()
        for i in range(nrows):
            row = Vector()
            for j in range(ncols):
                row.append(0)
            m.append(row)
        return m

    @staticmethod
    def random(nrows, ncols):
        m = Matrix()
        for i in range(nrows):
            row = Vector()
            for j in range(ncols):
                row.append(random.randint(-10, 10))
            m.append(row)
        return m

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, iterable):
        self.__data = []
        if not Vector._is_iterable(iterable):
            raise AttributeError('Must initialize matrix with an iterable.')
        for v in iterable:
            self.append(v)

    @property
    def max_col_norm(self):
        partial_max = 0
        for i in range(self.ncols):
            col_sum = sum(abs(x) for x in self.getcol(i))
            if col_sum > partial_max: partial_max = col_sum
        return partial_max

    @property
    def max_row_norm(self):
        partial_max = 0
        for i in range(self.nrows):
            row_sum = sum(abs(x) for x in self)
            if row_sum > partial_max: partial_max = row_sum
        return partial_max

    def norm(self, p):
        if p == float('inf'): return self.max_row_norm
        partial_sum = 0
        for i in range(self.nrows):
            for j in range(self.ncols):
                partial_sum += abs(self[i][j]) ** p
        return partial_sum**(1 / p)

    @property
    def nrows(self):
        return len(self)

    @property
    def ncols(self):
        try:
            return len(self[0])
        except IndexError:
            return 0

    def append(self, line):
        v = Vector(line)
        if self._valid_vector(v) or self.shape == (0,0):
            self.data.append(v)
        else:
            raise AttributeError('Can only append vectors with the same number of columns as the matrix.')

    def swap(self, i, j):
        aux = self[i]
        self[i] = self[j]
        self[j] = aux

    @property
    def shape(self):
        return self.nrows, self.ncols

    def getcol(self, j) -> Vector:
        col = Vector()
        for i in range(self.nrows):
            col.append(self[i][j])
        return col

    def transpose(self):
        t = Matrix()
        for i in range(self.ncols):
            t.append(self.getcol(i))
        return t

    def round(self, digits=4):
        for i in range(self.nrows):
            for j in range(self.ncols):
                self[i][j] = round(self[i][j], digits)
        return self

    def _valid_vector(self, vector):
        return type(vector) == Vector and (len(vector) == self.ncols)

    def _valid_matrix(self, matrix):
        return type(matrix) == Matrix and self.shape == matrix.shape

    def _valid_matrix_prod(self, matrix):
        return type(matrix) == Matrix and self.ncols == matrix.nrows

    property
    def square_matrix(self):
        return self.nrows == self.ncols

    def __init__(self, data=None):
        if data is None: data = []
        self.data = data

    def __iter__(self):
        for vector in self.data:
            yield vector

    def __getitem__(self, i):
        if type(i) == int:
            return self.data[i]
        elif type(i) == slice:
            return Matrix(self.data[i])
        else:
            raise NotImplementedError('Indexing on matrix is only supported with integers and simple slices.')

    def __setitem__(self, i, vector):
        v = Vector(vector)
        if self._valid_vector(v):
            self.data[i] = v
        else:
            raise AttributeError('All lines must have be vectors with the same number of lines.')

    def __repr__(self):
        return '[' + ',\n'.join(str(v) for v in self) + ']'

    def __add__(self, other):
        if not self._valid_matrix(other):
            raise exceptions.ArrayLengthError('Both matrices must have the same shape in a matrix sum.')
        msum = Matrix()
        for u, v in zip(self, other):
            msum.append(u + v)
        return msum

    def __len__(self):
        return self.data.__len__()

    def __neg__(self):
        neg = Matrix()
        for v in self:
            neg.append(-v)
        return neg

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        result = Matrix()
        if Vector._is_numeric(other):
            for v in self:
                result.append(other * v)
        elif self._valid_matrix_prod(other):
            for i in range(self.nrows):
                row = Vector()
                for j in range(other.ncols):
                    row.append(self[i] * other.getcol(j))
                result.append(row)
        elif not self._valid_matrix_prod(other):
            raise exceptions.ArrayLengthError('Left ncols must equal right nrows.')
        else:
            return NotImplemented
        return result

    def __rmul__(self, other):
        if Vector._is_numeric(other):
            return self * other
        else:
            return NotImplemented

    def __eq__(self, other):
        other = Matrix(other)
        if self.shape != other.shape: return False
        for r, s in zip(self, other):
            if r != s: return False
        return True

    def submatrix(self, removed_row, removed_col):
        # returns a submstrix after removing row and col
        new_matrix = Matrix()
        for i in range(self.nrows):
            if i == removed_row: continue
            row = Vector()
            for j in range(self.ncols):
                if j == removed_col: continue
                row.append(self[i][j])
            new_matrix.append(row)
        return new_matrix

    def det_recursive(self):
        if not self.square_matrix: raise exceptions.ArrayLengthError('Can only measure the determinant of a square matrix.')
        if self.nrows == 1: return self[0][0]
        partial_sum = 0
        for j, x in enumerate(self[0]):
            partial_sum += (-1)**j * x * self.submatrix(0, j).det_recursive()
        return partial_sum

    def det(self):
        l, u = self.lu_decomp()
        det = 1
        for i in range(self.nrows):
            det += u[i][i]
        return det

    def is_upper_trig(self):
        for i in range(self.nrows):
            for j in range(0, i):
                if self[i][j] != 0: return False
        return True

    def get_upper_trig(self):
        m = Matrix.zeros(self.nrows, self.ncols)
        for i in range(self.nrows):
            for j in range(i, self.ncols):
                m[i][j] = self[i][j]
        return m

    def _retroactive_substitutions(self, b:Vector):
        if type(b) != Vector:
            raise ValueError('Please convert independent coefficients to a Vector before trying to solve for it.')
        elif not self.square_matrix():
            raise exceptions.ArrayLengthError('You need a square matrix to solve any system.')
        if not self.is_upper_trig():
            raise exceptions.MatrixTypeError('You need an upper trig matrix to perform successive substitutions.')
        else:
            x = Vector([0] * len(b))
            N = self.nrows
            for i in range(N - 1, -1, -1):
                if self[i][i] == 0: raise exceptions.LinearDependencyError('Can\'t retroactively substitute on a singular matrix.')
                partial = b[i]
                for j in range(N - 1, i, -1):
                    partial -= (self[i][j] * x[j])
                x[i] = partial / self[i][i]
            return x

    def is_lower_trig(self):
        for i in range(self.nrows):
            for j in range(i+1, self.ncols):
                if self[i][j] != 0: return False
        return True

    def get_lower_trig(self):
        m = Matrix.zeros(self.nrows, self.ncols)
        for i in range(self.nrows):
            for j in range(0, i + 1):
                m[i][j] = self[i][j]
        return m

    def _successive_substitutions(self, b:Vector):
        if type(b) != Vector:
            raise ValueError('Please convert independent coefficients to a Vector before trying to solve for it.')
        elif not self.square_matrix():
            raise exceptions.ArrayLengthError('You need a square matrix to solve any system.')
        elif not self.is_lower_trig():
            raise exceptions.MatrixTypeError('You need a lower trig matrix to perform successive substitutions.')
        else:
            x = Vector()
            N = self.nrows
            for i in range(N):
                if self[i][i] == 0: raise exceptions.LinearDependencyError('Can\'t retroactively substitute on a singular matrix.')
                partial = b[i]
                for j in range(0, i):
                    partial -= (self[i][j] * x[j])
                x.append(partial / self[i][i])
            return x

    def lu_decomp(self):
        if not self.square_matrix():
            raise exceptions.ArrayLengthError('Can\'t decompose non-square matrix.')
        N = self.nrows
        LU = Matrix(self)
        p = Matrix.identity(N)
        for linha_pivotal in range(N):
            pivotal_col = LU.getcol(linha_pivotal)[linha_pivotal:]
            i, pivot = pivotal_col.max()
            if pivot == 0:
                # print('ERROR!')
                # print('Pivotal col:', pivotal_col)
                # print(self)
                # print(LU)
                # print('Pivot:', linha_pivotal)
                raise exceptions.LinearDependencyError('Can\'t decompose a singular matrix (det = 0).')
            i += linha_pivotal
            p.swap(linha_pivotal, i)
            LU.swap(linha_pivotal, i)
            for linha_atual in range(linha_pivotal + 1, N):
                mult = LU[linha_atual][linha_pivotal] / pivot
                LU[linha_atual][linha_pivotal] = mult
                for k in range(linha_pivotal + 1, N):
                    LU[linha_atual][k] -= mult * LU[linha_pivotal][k]
        L = LU.get_lower_trig()
        for i in range(N):
            L[i][i] = 1
        U = LU.get_upper_trig()
        return L, U, p

    # def lu_decomp(self):
    #     if not self.square_matrix():
    #         raise exceptions.ArrayLengthError('Can\'t decompose non-square matrix.')
    #     N = self.nrows
    #     U = Matrix(self)
    #     L = Matrix.identity(N)
    #     for linha_pivotal in range(N - 1):
    #         pivot = U[linha_pivotal][linha_pivotal]
    #         for linha_atual in range(linha_pivotal + 1, N):
    #             mult = U[linha_atual][linha_pivotal] / pivot
    #             L[linha_atual][linha_pivotal] = mult
    #             U[linha_atual] -= mult * U[linha_pivotal]
    #     return L, U.get_upper_trig()

    @staticmethod
    def _quicksolve(l, u, b):
        x = Matrix()
        for i in range(b.ncols):
            y = l._successive_substitutions(b.getcol(i))
            x.append(u._retroactive_substitutions(y))
        return x.transpose()

    def solve(self, b, max_error=0.005):
        if type(b) == Vector:
            b = Matrix([b]).transpose()
        elif type(b) == Matrix:
            pass
        else:
            raise NotImplementedError('Please insert a matrix or a vector of independent coefficients.')
        l, u, p = self.lu_decomp()
        x = self._quicksolve(l, u, p*b)
        r = self * x - b
        error = r.norm(2) / x.norm(2)
        while error > max_error:
            c = self._quicksolve(l, u, p*r)
            x += c
            r = self * x - b
            error = r.norm(2) / x.norm(2)
            print(error)
        return x

    def inv(self):
        if not self.square_matrix: raise exceptions.ArrayLengthError('Can only invert a square matrix.')
        return self.solve(Matrix.identity(self.nrows))
