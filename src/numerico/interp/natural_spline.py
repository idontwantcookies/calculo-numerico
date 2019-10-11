import numpy as np

from .core import CoreInterp


class NaturalSpline(CoreInterp):
    def _setUp(self):
        dely = np.zeros()
