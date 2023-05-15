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
# conn=np.array([[1,2,6,5], [2,3,7,6],[3,4,8,7],[5,6,10,9],[6,7,11,10],[7,8,12,11]])
# Changes in element connections depending on rectangular or triangular elements in anti clockwise direction
con=np.array([[1,2,5,4], [2,3,6,5],[4,5,8,7],[5,6,9,8]])
n_coord=np.array([[0,0.5,1,0,0.5,1,0,0.5,1], [0,0,0,0.5,0.5,0.5,1,1,1]])

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
NG =len(x_coord)
# print(NPE)
# print(NG)

k_glob=np.zeros((NG , NG))
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

# Changes in analytical solution
#Analytical_Solution
# u_ana = np.zeros(len(x_coord))
# for i in range(len(x_coord)):
# 	  u_ana[i] =((np.cosh(np.pi*y_coord[i]/6)*np.cos(np.pi*x_coord[i]/6))/(np.cosh(np.pi/3)))
u_ana = np.array([0.31071,0.24107,0.,0.24107,0.19286,0.,0.,0.,0.])
u_rec =u_ana
u_tri = np.array([0.3012,0.2032,0.,0.2326,0.1759,0.,0.,0.,0.])
u_rec16 = np.array([0.2984,0.2322,0.,0.2675,0.135,0.,0.,0.,0.])


# print(np.around(u_ana, decimals = 4))

#Shape_Functions
# a=0.5
# b=0.5
# Nf1=(1-(X/a))*(1-(Y/b));
# Nf2=((X/a))*(1-(Y/b));
# Nf3=((X/a))*((Y/b));
# Nf4=(1-(X/a))*((Y/b));

# step size of element in x direction a=0.5
# step size of element in y direction b=0.5

Nf1=0.25*(1-X)*(1-Y);
Nf2=0.25*(1+X)*(1-Y);
Nf3=0.25*(1+X)*(Y+1);
Nf4=0.25*(1-X)*(1+Y);



Nf=np.array([Nf1,Nf2,Nf3,Nf4]);


for k in range(n_ele):
    x=0.0
    y=0.0

# Elemental_Stiffness and Elemental Force Matrix
    if (k == 0):
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
        # Changes for boundary forces
        b = 1.

        for i in range(NPE):
            for j in range(NPE):
                jac = sym.diff(x, X)*sym.diff(y, Y)-sym.diff(x, Y)*sym.diff(y, X)  # defining jacobian
                # print (jac)
                func = ((sym.diff(Nf[i], X) * sym.diff(Nf[j], X))+(sym.diff(Nf[i], Y) * sym.diff(Nf[j], Y)))/jac

                func_2 = Nf[i] * b
                Gt = G_int.Integration(func, func_2, p, w, X,Y, jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness_2D()
                k_ele_f[i][j] = float(k_ele_f[i][j] + I)
            # print(6*k_ele_f)

            I2 = Gt.gauss_quad_force_2D()
            f_ele_f[i] =(f_ele_f[i]+I2)
        # print(16*f_ele_f)


    if (k != 0):
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
        b = 1.

        for i in range(NPE):
            for j in range(NPE):
                jac = sym.diff(x, X) * sym.diff(y, Y) - sym.diff(x, Y) * sym.diff(y, X)  # defining jacobian
                # print (jac)
                func = ((sym.diff(Nf[i], X) * sym.diff(Nf[j], X)) + (sym.diff(Nf[i], Y) * sym.diff(Nf[j], Y))) / jac
                func_2 = Nf[i] * b
                Gt = G_int.Integration(func, func_2, p, w, X, Y, jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness_2D()
                k_ele_f[i][j] = float(k_ele_f[i][j] + I)

            I2 = Gt.gauss_quad_force_2D()
            f_ele_f[i] = (f_ele_f[i] + I2)



    # Global_Stiffness and Force Matrix
    LM=conn
    LM_loc=LM[k]
    # LM_1=np.array([0,1,5,4]);
    # LM_2 = np.array([1,2,6,5]);


    for i in range(NPE):
        m = LM_loc[i]
        # r= LM_2[i]
        for j in range(NPE):
            n=LM_loc[j]
            # s = LM_2[j]
            k_glob[m][n] = k_glob[m][n] + k_ele_f[i][j]
            # k_glob[r][s] = k_glob[r][s]+ k_ele_s[i][j]
        f_glob[m] = f_glob[m] + f_ele_f[i]
        # f_glob[r] = f_glob[r] + f_ele_s[i]


# Neumann boundary condition
node_nb = np.array([1, 2, 4, 5])
node_nbc=node_nb-1
# Dirichlet boundary conditions
node_dbc = np.array([[3,6,7,8,9], [0.,0.,0.,0.,0.]])
# print(node_dbc)


for i in range(len(f_glob)):
    for m in range(len(node_nbc)):
        if node_nbc[m]  == i:
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
u = solveobjec.get_res()
# Printing solutions===================
print("Printing x coorinates of the given governing equation:\n", x_coord)
print("Printing y coorinates of the given governing equation:\n",y_coord)
print("Printing FEA solution of the given governing equation:\n", np.around(u, decimals = 4))
print("Printing Analytical solution of the given governing equation:\n", np.around(u_ana, decimals = 4))

# Plotting Analytical vs Fea Solution
# Error code===================
norm = 0.
error = np.zeros(len(x_coord))

u_ana=np.around(u_ana, decimals = 4)
for i in range(len(x_coord)):
    error[i] = abs(u[i] - u_ana[i])

rms = sqrt(mean_squared_error(u_ana, u))

# Error Printing===================
print("Printing error of the given governing equation:\n", np.around(error, decimals = 4))
print("Printing rms error of the given governing equation:\n", np.around(rms, decimals = 4))
plt.plot(x_coord, u_rec, 'r-') # plotting t, a separately
plt.plot(x_coord,u_tri , 'bo') # plotting t, b separately
plt.plot(x_coord, u_rec16, 'g+') # plotting t, c separately
plt.show()
# plot_objec = plot_res(xn,u,u_ana)  # initialize the object
# plot_objec.get_plot()


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(x_coord, y_coord, u_rec16,  linewidth=0, antialiased=False)
ax.set_title('Temperature  Plot for 16 rectangular elements', size=12)
ax.set_xlabel('X coordinates', size=12)
ax.set_ylabel('Y coordinates', size=12)
plt.show()






















