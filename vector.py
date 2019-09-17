import math
import pdb

import numpy as np

import exceptions


class Vector:
    @property
    def vector(self):
        return self.__vector

    @vector.setter
    def vector(self, iterable):
        if not self._is_iterable(iterable):
            raise AttributeError('Must initialize vector with an iterable.')
        self.__vector = list()
        for x in iterable:
            if not self._is_numeric(x):
                raise AttributeError('All iterable elements must be numeric.')
            else:
                self.__vector.append(x)

    def append(self, value):
        if type(value) not in (int, float):
            raise AttributeError('Can only append numeric values to a vector.')
        self.vector.append(value)

    def apply(self, func):
        v = Vector()
        for x in self:
            v.append(func(x))
        return v

    def angle(self, other):
        # uv = |u||v|cos(theta) => theta = acos(uv/|u||v|)
        other = Vector(other)
        dot = self * other
        cos = dot / (abs(self) * abs(other))
        return math.acos(cos)

    def max(self):
        index = 0
        value = 0
        for i, x in enumerate(self):
            if (abs(x)) > abs(value):
                value = x
                index = i
        return index, value

    def sum(self):
        return self.sumpow(1)

    def sumpow(self, power):
        if power == 0: return len(self)
        partial = 0
        for x in self:
            partial += x**power
        return partial

    @staticmethod
    def _is_numeric(value):
        attrs = ('__add__', '__sub__', '__mul__', '__truediv__', '__pow__')
        return all(hasattr(value, attr) for attr in attrs)

    @staticmethod
    def _is_iterable(iterable):
        try:
            iter(iterable)
        except TypeError:
            return False
        else:
            return True

    def _same_size(self, other):
        return len(self) == len(other)

    def __init__(self, vector=None):
        if vector is None: vector = list()
        self.vector = vector

    def __iter__(self):
        for x in self.vector:
            yield x

    def __repr__(self):
        return str(self.vector)

    def __len__(self):
        return self.vector.__len__()

    def __getitem__(self, i):
        if type(i) == int:
            return self.vector[i]
        elif type(i) == slice:
            return Vector(self.vector[i])

    def __setitem__(self, i, value):
        if not self._is_numeric(value):
            raise AttributeError('Can only set numeric values to a vector.')
        self.vector[i] = value

    def __add__(self, other_vector):
        if type(other_vector) is not Vector:
            return NotImplemented
        if not self._same_size(other_vector):
            raise exceptions.ArrayLengthError('Can only sum 2 vectors of the same legth.')
        sum_vector = Vector()
        for x, y in zip(self, other_vector):
            sum_vector.append(x + y)
        return sum_vector

    def __neg__(self):
        negative_vector = Vector()
        for x in self:
            negative_vector.append(-x)
        return negative_vector

    def __sub__(self, other_vector):
        return self + (-other_vector)

    def __mul__(self, other):
        # scalar product
        if self._is_numeric(other):
            product = Vector()
            for x in self:
                product.append(other * x)
        elif self._is_iterable(other) and self._same_size(other):
            product = 0
            for x, y in zip(self, other):
                product += x * y
        elif not self._same_size(other):
            pdb.set_trace()
            raise exceptions.ArrayLengthError('Both vectors must be equal length.')
        else:
            return NotImplemented
        return product

    def __rmul__(self, other):
        return self * other

    def __abs__(self):
        # returns the length of the vector
        sq_sum = 0
        for x in self:
            sq_sum += x * x
        return sq_sum ** 0.5

    def __eq__(self, other):
        other = Vector(other)
        if len(other) != len(self): return False
        for x, y in zip(self, other):
            if x != y: return False
        return True
