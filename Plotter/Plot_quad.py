import matplotlib.pyplot as pt

class plotting(object):
    '''Define Global Stiffness Matrix and Global Force Matrix.'''

    def __init__(self, xn=0., u=0.,u_ana=0.):
        self.xn = xn
        self.u = u
        self.u_ana = u_ana

    def get_plot(self):
        """ Solve the equation using spsolve """
        xn =  self.xn
        u = self.u
        u_ana = self.u_ana
        pt.plot(xn, u,label="u_fea")
        pt.plot(xn, u_ana,'*', label="u_ana")
        pt.legend(loc='best')
        pt.xlabel('x_coordinates')
        pt.ylabel('Displacement value')
        pt.title('U_ana vs U_fea for  quadratic elements')
        #pt.title('U_ana vs U_fea for 64 linear elements')
        pt.grid(True)
        pt.show()






