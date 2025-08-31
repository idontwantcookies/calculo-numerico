from src.interpolation.natural_spline import NaturalSpline


class NotAKnotSpline(NaturalSpline):
    def _set_arbitrary_s2(self):
        n = self.n
        self.s2[0] = ((self.h[0] + self.h[1]) * self.s2[1] -
                      self.h[0] * self.s2[2]) / self.h[1]
        self.s2[-1] = ((self.h[-1] + self.h[-2]) * self.s2[-1] -
                       self.h[-1] * self.s2[-2]) / self.h[-2]

    def _build_matrix(self):
        m = super()._build_matrix()
        m[0, 0] = (self.h[0] + self.h[1]) * \
            (self.h[0] + 2 * self.h[1]) / self.h[1]
        m[0, 1] = (self.h[1]**2 - self.h[0]**2) / self.h[1]
        m[-1, -2] = (self.h[-2]**2 - self.h[-1]**2) / self.h[-2]
        m[-1, -1] = (self.h[-1] + self.h[-2]) * \
            (self.h[-1] + 2 * self.h[-2]) / self.h[-2]
        return m
