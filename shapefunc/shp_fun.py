import numpy as np


class Lagsf(object):
    """Evaluates Lagrange interpolation functions and its derivatives. Degree of polynomial is d =1,2,3
    Example: K=Lagsf()
             F=Lagsf.F(0.2,1)"""
    def __init__(self, X, d):
        self.X = X
        self.d = d

    def f(self):
        ''' Evaluates 1-self.dimensional Lagrange shape functions
        of self.degree self.d (self.d=1,2,3) at point self.X '''
        if self.d == 1:
            sf = np.array([0.5 * (1 - self.X), 0.5 * (1 + self.X)]) # Linear Element
        if self.d == 2:
            sf = np.array([self.X * (0.500000000000000 * self.X - 0.500000000000000), -self.X ** 2 + 1,
                           0.500000000000000 * (self.X + 1) * self.X])
        if self.d == 3:
            sf = np.array([-9 * (self.X - 1) * (self.X - 1 / 3) * (self.X + 1 / 3) / 16,
                           27 * (self.X - 1) * (self.X - 1 / 3) * (self.X + 1) / 16,
                           -27 * (self.X - 1) * (self.X + 1 / 3) * (self.X + 1) / 16,
                           9 * (self.X - 1 / 3) * (self.X + 1 / 3) * (self.X + 1) / 16])
        return sf

    def df(self):
        ''' Evaluates First self.derivative of 1-self.dimensional Lagrange shape functions
        of self.degree self.d(1,2,3) at point self.X '''
        if self.d == 1:
            dsf = np.array([-0.5, 0.5])
        if self.d == 2:
            dsf = np.array([1.00000000000000 * self.X - 0.500000000000000, -2 * self.X,
                            1.00000000000000 * self.X + 0.500000000000000])
        if self.d == 3:
            dsf = np.array([-9 * (self.X - 1) * (self.X - 1 / 3) / 16 - 9 * (self.X - 1) * (self.X + 1 / 3) / 16 - 9 * (
                        self.X - 1 / 3) * (self.X + 1 / 3) / 16,
                            27 * (self.X - 1) * (self.X - 1 / 3) / 16 + 27 * (self.X - 1) * (self.X + 1) / 16 + 27 * (
                                        self.X - 1 / 3) * (self.X + 1) / 16,
                            -27 * (self.X - 1) * (self.X + 1 / 3) / 16 - 27 * (self.X - 1) * (self.X + 1) / 16 - 27 * (
                                        self.X + 1 / 3) * (self.X + 1) / 16,
                            9 * (self.X - 1 / 3) * (self.X + 1 / 3) / 16 + 9 * (self.X - 1 / 3) * (
                                        self.X + 1) / 16 + 9 * (self.X + 1 / 3) * (self.X + 1) / 16])
        return dsf
