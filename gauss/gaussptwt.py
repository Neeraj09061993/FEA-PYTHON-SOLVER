from numpy.polynomial.legendre import leggauss as gauss
import math as ma
from numpy import pi
import numpy as np
from shapefunc.shp2dtriO4 import Lagsf2


class Gauss1d:
    """ Gauss points and Gauss weights for 1d element"""

    def __init__(self, gp=1):
        self.gp = gp

    def point_weight(self):
        r = self.gp

        '''Returns the Gauss points for a polynomial of degree 2*r-1'''


        pt = gauss(r)[0]
        wt = gauss(r)[1]

        return [pt, wt]


class Gauss2d(Gauss1d):
    """ Gauss points and Gauss weights for 1d element"""

    def __init__(self, gp=1):
        super().__init__(gp)
        self.ptwt = super().point_weight()

    def point_weight(self):
        gp = self.gp
        pt = self.ptwt[0]
        wt = self.ptwt[1]
        return [pt, wt]


class gausstri(object):

    def __init__(self, gp=1):
        self.gp = gp

    def point_weight(self):
        r = self.gp
        if r == 1:
            pt = np.array([[1 / 3, 1 / 3]])
            wt = np.array([1])
        if r == 2:
            pt = np.array([[2 / 3, 1 / 6], [1 / 6, 1 / 6], [1 / 6, 2 / 3]])
            wt = np.array([1 / 3, 1 / 3, 1 / 3])
        if r == 3:
            p6 = 0.60000000000000000000000
            p2 = 0.20000000000000000000000
            pt = np.array([[1 / 3, 1 / 3], [p6, p2], [p2, p6], [p2, p2]])
            wt = np.array([-27 / 48, 25 / 48, 25 / 48, 25 / 48])
        if r == 4:
            a1 = 0.797426985353
            b1 = 0.101286507323
            a2 = 0.059715871789
            b2 = 0.470142064105
            w1 = 0.225000000000
            w2 = 0.125939180544
            w3 = 0.132394152788
            pt = np.array([[1 / 3, 1 / 3], [a1, b1], [b1, a1], [b1, b1], [a2, b2], [b2, a2], [b2, b2]])
            wt = np.array([w1, w2, w2, w2, w3, w3, w3])

        return [pt, wt]

class gausstri_new(object):

    def __init__(self, gp=1):
        self.gp = gp

    def point_weight(self):
        r = self.gp
        if r == 1:
            pt = np.array([[1 / 3, 1 / 3]])
            wt = np.array([1])
        if r == 2:
            pt = np.array([[2 / 3, 1 / 6], [1 / 6, 1 / 6], [1 / 6, 2 / 3]])
            wt = np.array([1 / 3, 1 / 3, 1 / 3])
        if r == 3:
            p6 = 0.60000000000000000000000
            p2 = 0.20000000000000000000000
            pt = np.array([[1 / 3, 1 / 3 , 1/3], [p2,p2,p6], [ p6,p2, p2], [p2,p6, p2]])
            wt = np.array([-27 / 48, 25 / 48, 25 / 48, 25 / 48])
        if r == 4:
            a1 = 0.797426985353
            b1 = 0.101286507323
            a2 = 0.059715871789
            b2 = 0.470142064105
            w1 = 0.225000000000
            w2 = 0.125939180544
            w3 = 0.132394152788
            pt = np.array([[1 / 3, 1 / 3], [a1, b1], [b1, a1], [b1, b1], [a2, b2], [b2, a2], [b2, b2]])
            wt = np.array([w1, w2, w2, w2, w3, w3, w3])

        return [pt, wt]
# if __name__ == "__main__":
#     K = gausstri(gp=4)
#     J = K.point_weight()
#     pt = J[0]
#     wt = J[1]
#     tp = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]])
#     sum = 0.
#
#     for (c, i) in enumerate(pt):
#         shp = Lagsf2(i[0], i[1], 2)
#         x = 0.
#         y = 0.
#         fu = shp.f()
#         print("fu", fu)
#         for (j, point) in enumerate(tp):
#             x = x + point[0] * fu[j]
#             y = y + point[1] * fu[j]
#         print("X", x)
#         print("Y[wt]", y, wt[c])
#         sum = sum + wt[c] * y * x
#     print(0.5 * sum)
