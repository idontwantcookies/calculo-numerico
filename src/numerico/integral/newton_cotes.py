import numpy as np

coefs = {
    1: np.array([1, 1]),
    2: np.array([1, 4, 1]),
    3: np.array([1, 3, 3, 1]),
    4: np.array([7, 32, 12, 32, 7, 1]),
    5: np.array([19, 75, 50, 50, 75, 19]),
    6: np.array([41, 216, 27, 272, 27, 216, 41]),
    7: np.array([751, 3577, 1323, 2989, 2989, 1323, 3577, 751]),
    8: np.array([989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989]),
}

class NewtonCotes:
    def __init__(self):
        pass
