class Integration(object):
    def __init__(self, func , func_2 ,p, w , X , Y , jac):
        self.func = func
        self.func2 = func_2
        self.p = p
        self.w = w
        self.X = X
        self.Y = Y
        self.jac = jac

    def gauss_quad_stiffness(self):
        func = self.func
        p= self.p
        w= self.w
        X = self.X
        jac = self.jac
        I = 0.0
        for i in range(len(p)):
            func = self.func
            fun = func.subs({X: p[i]})
            # jaco = jac.subs(X, p[i])
            I = I + w[i] * fun * jac
        return [I]

    def triangleQuad(self):
        func = self.func
        p= self.p
        w= self.w
        X = self.X
        jac = self.jac
        I = 0.0
        for i in range(len(p)):
            func = self.func
            fun = func.subs({X: x[i]})
            # jaco = jac.subs(X, p[i])
            I = I + w[i] * fun * jac
        return [I]

    def gauss_quad_stiffness_2D(self):
        func = self.func
        p= self.p
        w= self.w
        X = self.X
        Y = self.Y
        jac = self.jac
        I = 0.0
        for i in range(len(p)):
            for j in range(len(p)):
                func = self.func
                fun = w[i] * w[j] * func.subs({Y: p[j], X: p[i]})
                jaco = jac.subs({Y: p[j], X: p[i]})
                I = I + fun * jaco
        return [I]

    def gauss_quad_force(self):
        func2 = self.func2
        p= self.p
        w= self.w
        X = self.X
        jac = self.jac
        I2= 0.0
        for i in range(len(p)):
            func2 = self.func2
            fun2 = func2.subs({X: p[i]})
            # jaco2 = jac.subs(X, p[i])
            I2 = I2 + w[i] * fun2 * jac
        return [I2]

    def gauss_quad_force_2D(self):
        func2 = self.func2
        p= self.p
        w= self.w
        X = self.X
        Y = self.Y
        jac = self.jac
        I2= 0.0
        for i in range(len(p)):
            for j in range(len(p)):
                func2 = self.func2
                fun2 = w[i] * w[j] * func2.subs({Y: p[j], X: p[i]})
                jaco2 = jac.subs({Y: p[j], X: p[i]})
                I2 = I2 + fun2 * jaco2
        return [I2]


class Integration_new(object):

    def __init__(self, func, func_2, p, w, X,Y,x_p,y_p, jac) :

        self.func = func
        self.func2 = func_2
        self.p = p
        self.x_p = x_p
        self.y_p = y_p

        self.w = w
        self.X = X
        self.Y = Y
        self.jac = jac

    def trianglestiff(self):
        func = self.func
        x_p = self.x_p
        y_p = self.y_p
        w = self.w
        X = self.X
        Y = self.Y
        jac = self.jac
        I = 0.0
        for i in range(len(x_p)):
            for j in range(len(y_p)):
                func = self.func
                fun = w[i] * w[j] * func.subs({Y: y_p[j], X: x_p[i]})
                jaco = jac.subs({Y: y_p[j], X: x_p[i]})
                I = I + fun
        return I

    def triangleforce(self):
        func2 = self.func2
        x_p = self.x_p
        y_p = self.y_p
        w = self.w
        X = self.X
        Y = self.Y
        jac = self.jac
        I2 = 0.0
        for i in range(len(x_p)):
            for j in range(len(y_p)):
                func2 = self.func2
                fun2 = w[i] * w[j] * func2.subs({Y: y_p[j], X: x_p[i]})
                jaco2 = jac.subs({Y: y_p[j], X: x_p[i]})
                I2 = I2 + fun2
        return I2

class Integration_1(object):
    def __init__(self, func,func2,p,w,X,jac):
        self.func = func
        self.func2 = func2
        self.p = p
        self.w = w
        self.X = X
        self.jac = jac

    def gauss_quad_stiffness(self):
        func = self.func
        pt= self.p
        wt= self.w
        X = self.X
        jac = self.jac
        I = 0.0
        for i in range(len(pt)):
            fun = func.subs({X: pt[i]})
            jaco = jac.subs(X, pt[i])
            I = I + wt[i] * fun * jaco
        return [I]

    def gauss_quad_force(self):
        func2 = self.func2
        pt= self.p
        wt= self.w
        X = self.X
        jac = self.jac
        I2= 0.0
        for i in range(len(pt)):
            fun2 = func2.subs({X: pt[i]})
            jaco2 = jac.subs(X, pt[i])
            I2 = I2 + wt[i] * fun2 * jaco2
        return [I2]