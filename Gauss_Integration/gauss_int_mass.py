class Integration(object):
    def __init__(self, stiff_func,mass_func,force_func, p, w, X, jac):
        self.stiff_func = stiff_func
        self.mass_func = mass_func
        self.force_func = force_func
        self.p = p
        self.w = w
        self.X = X
        self.jac = jac

    def gauss_quad_stiffness(self):
        stiff_func = self.stiff_func
        pt= self.p
        wt= self.w
        X = self.X
        jac = self.jac
        I = 0.0
        for i in range(len(pt)):
            stiff_func = self.stiff_func
            stiff_func = stiff_func.subs({X: pt[i]})
            # jaco = jac.subs(X, pt[i])
            I = I + wt[i] * stiff_func * jac
        return [I]
    
    def gauss_quad_mass(self):
        mass_func = self.mass_func
        M=mass_func
        pt= self.p
        wt= self.w
        X = self.X
        jac = self.jac
        I3 = 0.0
        for i in range(len(pt)):
            mass_func = self.mass_func
            mass_func = mass_func.subs({X: pt[i]})
            # jaco = jac.subs(X, pt[i])
            I3 = I3 + wt[i] * mass_func * jac
        return [I3]


    def gauss_quad_force(self):
        force_func = self.force_func
        pt= self.p
        wt= self.w
        X = self.X
        jac = self.jac
        I2= 0.0
        for i in range(len(pt)):
            force_func = self.force_func
            force_func = force_func.subs({X: pt[i]})
            # jaco2 = jac.subs(X, pt[i])
            I2 = I2 + wt[i] * force_func * jac
        return [I2]
