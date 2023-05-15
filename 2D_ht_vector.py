import numpy as np
import sympy as sym
# from scipy.special.orthogonal import p_roots
from shapefunc.shp_fun import Lagsf
from mesh.readmsh import readmesh as line
from sklearn.metrics import mean_squared_error
from math import sqrt
from Solver.Solver import solution as res
from Elements.onedsecondorderode.coefficients import coefficients as eaproperties
from Plotter.Plot import plotting as plot_res
from scipy import sparse
from gauss import gaussptwt as GS
from Gauss_Integration import gauss_int  as G_int
import matplotlib.pyplot as plt
from Elements.onedsecondorderode.secondorderode import odeelement as bar

"""Calculates the element stiffness  matrix and heat generated  contributions for a steady heat transfer problem,
        governed by the differential equation d/dx(EAd^u/dx)=B. Replace E=1., A=1.,B=-1. with appropriate
         quantities to solve  problems of the differential equation."""

# Find Gauss Points and Gauss Weight

NGP=2
G = GS.Gauss1d(NGP)
p, w = G.point_weight()
# print(w)
# print(p)






# #=====Solve using Linear Finite Element ==============================================================================================

# filename = "mesh\square_mesh_2D.msh"
# Mesh_obj = line(filename=filename)
# Mesh_obj.readmshfile()
# mesh=Mesh_obj.mesh
# meshinfo=Mesh_obj.info
# physical=Mesh_obj.physical
# #"""Defining_Nodes"""
# n_ele=meshinfo['Quads']
# conn=mesh['Full_Connectivity_dim']
# Changes in element connections depending on rectangular or triangular elements in anti clockwise direction

con=np.array([[1,3,5,7],[2,4,6,8],[2,4,6,8],[1,3,5,7]])
n_coord=np.array([[0,0,120,120,120,120,0,0], [0,0,0,0,160,160,160,160]])

conn=con-1

# Number of elements
n_ele=(len(conn))
# print(n_ele)

x_coord=(n_coord)[0, :]

y_coord=(n_coord)[1, :]

xn=np.sort((n_coord)[:, 0])

# print(np.around(u_bc, decimals = 4))


# """Nodes per Element"""
NPE = len((conn)[0, :])
NG = len(x_coord)
# print(NPE)
# print(NG)


k_glob=np.zeros((NG, NG))
f_glob=np.zeros((NG,1))

X=sym.Symbol('X')
Y=sym.Symbol('Y')

# d=1.0
# """Defining Coefficients of differential equations such as E value is 2*10^11 , A value is 30*10^-6 ,B value is 0 ,"""
#
coeff_obj = eaproperties(nel=n_ele, e=1., a=1., b=0.,bf=0.)
coeffs = coeff_obj.get_coeffs()
E=(coeffs['e'])
A=(coeffs['a'])

# #Analytical_Solution
# u_ana = np.zeros(len(x_coord))
# for i in range(len(x_coord)):
# 	  u_ana[i] =((np.cosh(np.pi*y_coord[i]/6)*np.cos(np.pi*x_coord[i]/6))/(np.cosh(np.pi/3)))
u_ana = 10**-4*np.array([0.,11.291,10.113,0.])

v_ana= 10**-4*np.array([0.,1.964,-1.080,0.])
#Shape_Functions
h=0.036
he=120
ke=160


c11=32*10**6
c12=8*10**6
c22=32*10**6
c66=12*10**6
F0=10*ke*0.5

Nf1=0.25*(1-X)*(1-Y);
Nf2=0.25*(1+X)*(1-Y);
Nf3=0.25*(1+X)*(Y+1);
Nf4=0.25*(1-X)*(1+Y);

# Nf1=X;
# Nf2=1-X-Y;
# Nf3=Y;

# Nf1 = (1-X/he);
# Nf2 = (1/he)*X-(1/ke)*Y;
# Nf3 = Y/ke;
# Ns1 = (1-Y/ke);
# Ns2 =X/he;
# Ns3 =(1/ke)*Y-(1/he)*X;




Nf=np.array([Nf1,Nf2,Nf3,Nf4]);
# Ns=np.array([Ns1,Ns2,Ns3]);


for k in range(n_ele):
    x=0.0
    y=0.0

# Elemental_Stiffness and Elemental Force Matrix
    if (k == 0 ):
        xn = x_coord[conn[k]]
        yn = y_coord[conn[k]]

        #xn = n_coord[:, 0]
        #yn = n_coord[:, 1]
        for i in range(NPE):
            # x_coor = float((n_coord)[0, i])
            # x = x + Nf[i] * x_coor


            x = x + Nf[i] * xn[i]

            # y_coor = float((n_coord)[1, i])
            # y = y + Nf[i] * y_coor
            y = y + Nf[i] * yn[i]


        k_ele_f = np.zeros((NPE, NPE))
        f_ele_f = np.zeros((NPE, 1))
        k_ele_s = np.zeros((NPE, NPE))
        f_ele_s = np.zeros((NPE, 1))
        b = 0.

        for i in range(NPE):
            for j in range(NPE):
                jac = sym.diff(x, X)*sym.diff(y, Y)-sym.diff(x, Y)*sym.diff(y, X)  # defining jacobian
                print(jac)
                func = h*(c11*(sym.diff(Nf[i], X) * sym.diff(Nf[j], X)) +(c66*(sym.diff(Nf[i], Y) * sym.diff(Nf[j], Y)))) / jac

                # func = (.25*he * he) * (ke/he)*(1 + X)* h* ((c11*(sym.diff(Nf[i], X) * sym.diff(Nf[j], X))) + (c66*(sym.diff(Nf[i], Y) * sym.diff(Nf[j], Y))))/jac

                func_2 = Nf[i] * b
                # print(func)
                Gt = G_int.Integration(func, func_2, p, w, X, Y, jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness_2D()
                k_ele_f[i][j] = float(k_ele_f[i][j] + I)
            # print(k_ele_f)
            I2 = Gt.gauss_quad_force_2D()

            f_ele_f[i] =(f_ele_f[i]+I2)
    # print(k_ele_f)


    if (k == 1 ):
        xn = x_coord[conn[k]]
        yn = y_coord[conn[k]]

        # xn = n_coord[:, 0]
        # yn = n_coord[:, 1]
        for i in range(NPE):
            # x_coor = float((n_coord)[0, i])
            # x = x + Nf[i] * x_coor

            x = x + Nf[i] * xn[i]

            # y_coor = float((n_coord)[1, i])
            # y = y + Nf[i] * y_coor
            y = y + Nf[i] * yn[i]


        k_ele_f = np.zeros((NPE, NPE))
        f_ele_f = np.zeros((NPE, 1))
        k_ele_s = np.zeros((NPE, NPE))
        f_ele_s = np.zeros((NPE, 1))
        b = 0.

        for i in range(NPE):
            for j in range(NPE):
                jac = sym.diff(x, X) * sym.diff(y, Y) - sym.diff(x, Y) * sym.diff(y, X)  # defining jacobian
                # print(jac)

                func = h * ((c12*(sym.diff(Nf[i], X) * sym.diff(Nf[j], Y))) + (c66*(sym.diff(Nf[i], Y) * sym.diff(Nf[j], X))))/jac
                func_2 = Nf[i] * b
                Gt = G_int.Integration(func, func_2, p, w, X, Y, jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness_2D()
                k_ele_f[i][j] = float(k_ele_f[i][j] + I)
                # print(k_ele_f)
            I2 = Gt.gauss_quad_force_2D()

            f_ele_f[i] = (f_ele_f[i] + I2)
        # print(k_ele_f)

    # print(k_ele_f)
    if (k == 2 ):
        xn = x_coord[conn[k]]
        yn = y_coord[conn[k]]

        # xn = n_coord[:, 0]
        # yn = n_coord[:, 1]
        for i in range(NPE):
            # x_coor = float((n_coord)[0, i])
            # x = x + Nf[i] * x_coor

            x = x + Nf[i] * xn[i]

            # y_coor = float((n_coord)[1, i])
            # y = y + Nf[i] * y_coor
            y = y + Nf[i] * yn[i]


        k_ele_f = np.zeros((NPE, NPE))
        f_ele_f = np.zeros((NPE, 1))
        k_ele_s = np.zeros((NPE, NPE))
        f_ele_s = np.zeros((NPE, 1))
        b = 0.

        for i in range(NPE):
            for j in range(NPE):
                jac = sym.diff(x, X) * sym.diff(y, Y) - sym.diff(x, Y) * sym.diff(y, X)  # defining jacobian
                # print(jac)

                func =  h* ((c12*(sym.diff(Nf[j], X) * sym.diff(Nf[i], Y))) + (c66*(sym.diff(Nf[j], Y) * sym.diff(Nf[i], X))))/jac
                func_2 = Nf[i] * b
                Gt = G_int.Integration(func, func_2, p, w, X, Y, jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness_2D()
                k_ele_f[i][j] = float(k_ele_f[i][j] + I)
                # print(k_ele_f)
            I2 = Gt.gauss_quad_force_2D()

            f_ele_f[i] = (f_ele_f[i] + I2)
        # print(k_ele_f)

    # print(k_ele_f)
    if (k == 3 ):
        xn = x_coord[conn[k]]
        yn = y_coord[conn[k]]

        # xn = n_coord[:, 0]
        # yn = n_coord[:, 1]
        for i in range(NPE):
            # x_coor = float((n_coord)[0, i])
            # x = x + Nf[i] * x_coor

            x = x + Nf[i] * xn[i]

            # y_coor = float((n_coord)[1, i])
            # y = y + Nf[i] * y_coor
            y = y + Nf[i] * yn[i]


        k_ele_f = np.zeros((NPE, NPE))
        f_ele_f = np.zeros((NPE, 1))
        k_ele_s = np.zeros((NPE, NPE))
        f_ele_s = np.zeros((NPE, 1))
        b = 0.

        for i in range(NPE):
            for j in range(NPE):
                jac = sym.diff(x, X) * sym.diff(y, Y) - sym.diff(x, Y) * sym.diff(y, X)  # defining jacobian
                # print(jac)

                func = h * ((c66*(sym.diff(Nf[i], X) * sym.diff(Nf[j], X))) + (c22*(sym.diff(Nf[i], Y) * sym.diff(Nf[j], Y))))/jac
                func_2 = Nf[i] * b
                Gt = G_int.Integration(func, func_2, p, w, X, Y, jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness_2D()
                k_ele_f[i][j] = float(k_ele_f[i][j] + I)
                # print(k_ele_f)
            I2 = Gt.gauss_quad_force_2D()

            f_ele_f[i] = (f_ele_f[i] + I2)
        # print(k_ele_f)

    # print(k_ele_f)
    # Global_Stiffness and Force Matrix
    LM = conn
    LM_loc = LM[k]
    LM_1 = LM[0];
    LM_2 = LM[1];

    if(k==0):


        for i in range(NPE):
            m = LM_1[i]
            # r= LM_2[i]
            for j in range(NPE):
                n=LM_1[j]
                # s = LM_2[j]
                # print(k_ele_f)
                k_glob[m][n] = k_glob[m][n] + k_ele_f[i][j]
                # print(k_glob)
                # k_glob[r][s] = k_glob[r][s]+ k_ele_s[i][j]
            f_glob[m] = f_glob[m] + f_ele_f[i]
            # f_glob[r] = f_glob[r] + f_ele_s[i]
    if (k == 1 ):


        for i in range(NPE):
            m = LM_1[i]
            # r= LM_2[i]
            for j in range(NPE):
                n = LM_2[j]
                # s = LM_2[j]
                # print(k_ele_f)
                k_glob[m][n] = k_glob[m][n] + k_ele_f[i][j]
                # print(k_glob)
                # k_glob[r][s] = k_glob[r][s]+ k_ele_s[i][j]
            f_glob[m] = f_glob[m] + f_ele_f[i]
            # f_glob[r] = f_glob[r] + f_ele_s[i]
    if (k == 2 ):

        for i in range(NPE):
            m = LM_2[i]
            # r= LM_2[i]
            for j in range(NPE):
                n = LM_1[j]
                # s = LM_2[j]
                # print(k_ele_f)
                k_glob[m][n] = k_glob[m][n] + k_ele_f[i][j]
                # print(k_glob)
                # k_glob[r][s] = k_glob[r][s]+ k_ele_s[i][j]
            f_glob[m] = f_glob[m] + f_ele_f[i]
            # f_glob[r] = f_glob[r] + f_ele_s[i]
    if (k == 3):

        for i in range(NPE):
            m = LM_2[i]
            # r= LM_2[i]
            for j in range(NPE):
                n = LM_2[j]
                # s = LM_2[j]
                print(k_ele_f)
                k_glob[m][n] = k_glob[m][n] + k_ele_f[i][j]
                print(k_glob)
                # k_glob[r][s] = k_glob[r][s]+ k_ele_s[i][j]
            f_glob[m] = f_glob[m] + f_ele_f[i]
            # f_glob[r] = f_glob[r] + f_ele_s[i]

    # Neumann boundary condition
print(k_glob)
node_nb = np.array([[3,5], [F0,F0]])
node_nbc=node_nb-1

node_dbc = np.array([[1,2,7,8], [0.,0.,0.,0.]])
# print(node_dbc)


for i in range(len(f_glob)):
    for m in range(len(node_nbc)):
        if node_nbc[0][m] == i:
            f_glob[i] = f_glob[i] + F0
            for n in range(len(node_dbc[0])):
                f_glob[i] = f_glob[i] - (k_glob[i][int(node_dbc[0][n]) - 1] * node_dbc[1][n])



    # Dirichlet boundary conditions
    for m in range(len(node_dbc[0])):
        for i in range(len(k_glob[0])):
            if node_dbc[0][m] - 1 == i:
                for j in range(len(k_glob[0])):
                    f_glob[j] = f_glob[j] - k_glob[j][int(node_dbc[0][m]) - 1] * node_dbc[1][m]
                    if i != j:
                        k_glob[i][j] = 0.0
                        k_glob[j][i] = 0.0
                    else:
                        k_glob[i][j] = 1.0
                f_glob[i] = node_dbc[1][m]

# solving equations     ===================
# print(k_glob)
# print(f_glob)
solveobjec = res(sparse.csr_matrix(k_glob), f_glob)  # initialize the object
u = solveobjec.get_res()# Printing solutions===================
print("Printing x coorinates of the given governing equation:\n", x_coord)
print("Printing x coorinates of the given governing equation:\n",y_coord)
print("Printing FEA solution of the given governing equation:\n", np.around(u, decimals = 9))
print("Printing Analytical solution of the given governing equation:\n", np.around(u_ana, decimals = 9))
print("Printing Analytical solution of the given governing equation:\n", np.around(v_ana, decimals = 9))


# Plotting Analytical vs Fea Solution
# Error code===================
norm = 0.
error = np.zeros(len(x_coord))
# u_ana = np.array([0.6128,0.5307,0.3064,0,0.7030,0.6088,0.3515,0,0,0,0,0])
u_ana=np.around(u_ana, decimals = 9)
u_odd=u[::2]
u_even=u[1::2]
x_odd=x_coord[::2]
y_odd=y_coord[::2]
x_even=x_coord[1::2]
y_even=y_coord[1::2]
u_ana_odd = u_ana[::2]
u_ana_even=u_ana[1::2]

rms_e = sqrt(mean_squared_error(u_ana, u_odd))
rms_o = sqrt(mean_squared_error(v_ana, u_even))


# Error Printing===================
print("Printing error of the given governing equation:\n", np.around(error, decimals = 4))
print("Printing rms error of the given governing equation:\n", np.around(rms_e, decimals = 4))
print("Printing rms error of the given governing equation:\n", np.around(rms_o, decimals = 4))



fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(x_odd, y_odd, u_odd,  linewidth=0, antialiased=False)
ax.set_title('Displacement in x direction', size=12)
ax.set_xlabel('X ', size=12)
ax.set_ylabel('Y ', size=12)
ax.set_zlabel('U ', size=12)
plt.show()

fig2 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(x_even, y_even, u_even,  linewidth=0, antialiased=False)
ax.set_title('Displacement in y direction', size=12)
ax.set_xlabel('X ', size=12)
ax.set_ylabel('Y ', size=12)
ax.set_zlabel('V', size=12)
plt.show()






