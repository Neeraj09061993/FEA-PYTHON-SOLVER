from scipy.sparse.linalg import spsolve
from scipy import sparse


class solution(object):
    '''Define Global Stiffness Matrix and Global Force Matrix.'''

    def __init__(self, k_glob=0, f_glob=0.,):
        self.k_glob = k_glob
        self.f_glob = f_glob


    def get_res(self):
        """ Solve the equation using spsolve """
        k_glob =  self.k_glob
        f_glob = self.f_glob
        u=spsolve(k_glob, f_glob)
        u = spsolve(sparse.csr_matrix(k_glob),f_glob)
        return u





