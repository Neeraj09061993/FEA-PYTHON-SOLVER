import numpy as np
from math import sin


class coefficients(object):
    '''Define necessary coefficients of the differential equations'''

    def __init__(self, nel=0, e=1., a=1., b=0., bf=0., m=0., time=0):
        self.nel = nel
        self.e = e
        self.a = a
        self.b = b
        self.m = m
        self.bf = bf
        self.time = time

    def get_coeffs(self):
        """ Define elementwise constant coefficients A(Area), E (Young's Modulus) and B (Surface convection type) and (Source term type) BF coefficients for bar(
        type) element. Also, define them as functions of time if needed. """
        nel = self.nel
        ones = np.ones(nel)
        t = self.time
        e = self.e
        a = self.a
        b = self.b * ones
        bf = self.bf * ones
        m = self.m * ones

        # ''''''''''''''''''''''''''''''''''''''''''''''''
        # Place any specific definitions here'''
        # ''''''''''''''''''''''''''''''''''''''''''''''''
        # definitions of material properties for the Reddy problem 4.3.1
        # e[0:1] = 70
        # e[1:2] = 40
        # e[2:3] = 20
        # ''''''''''''''''''''''''''''''''''''''''''''''''
        coeffs = {'e': e, 'a': a, 'b': b, 'bf': bf, 'm': m}

        return coeffs

# if __name__ == '__main__':
#      K = coefficients(nel=85)
#      print(K.get_coeffs()['e'])
