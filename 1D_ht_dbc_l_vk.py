import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt
import sympy as sym
from shapefunc.shp_fun import Lagsf
from mesh.readmsh import readmesh as line
from Solver.Solver import solution as res
from Elements.onedsecondorderode.coefficients import coefficients as eaproperties
from Plotter.Plot_heat import plotting as plot_res
from scipy import sparse
from gauss import gaussptwt as GS
from Gauss_Integration import gauss_int as G_int

"""Calculates the element stiffness matrix and heat generated  contributions for a steady heat transfer problem,
        governed by the differential equation d/dx(KAd^u/dx)+Aq=B. Replace K,A,B. with appropriate
         quantities to solve  problems of the differential equation. Problem 4.3.1 JN Reddy book"""

# Find Gauss Points and Gauss Weight

NGP = 3
G = GS.Gauss1d(NGP)

p, w = G.point_weight()
# print(w)
# print(p)

# =====Solve using Linear Finite Element ==============================================================================================
""" 39 linear elements mesh file"""
#filename = "mesh\Line_39_element.msh"

""" Three linear elements mesh file"""
#filename = "mesh\Line_3_element_cond.msh"

""" Three quadratic elements mesh file"""
#filename = "mesh\Line_3_quad_element_cond.msh"

""" Three cubic elements mesh file"""
filename = "mesh\Line_3_cubic_element_cond.msh"


Mesh_obj = line(filename=filename)
Mesh_obj.readmshfile()
mesh = Mesh_obj.mesh
meshinfo = Mesh_obj.info
physical = Mesh_obj.physical
"""Defining_Nodes"""
conn = mesh['Connectivity']['Lines']
n_ele = meshinfo['Num_elem']

n_coord = mesh['Coords']
xn = np.sort((n_coord)[:, 0])
# x_c=np.linspace(0.,1.5,40)
# print(x_c)
"""Nodes per Element"""
NPE = meshinfo['Nodes_per_elem']
size = meshinfo['Num_nodes']


k_glob = np.zeros((size, size))
f_glob = np.zeros((size, 1))

X = sym.Symbol('X')



coeff_obj = eaproperties(nel=n_ele, e=1., a=1., b=1.)
coeffs = coeff_obj.get_coeffs()
K = (coeffs['e'])  # Thermal Conductivity
A = (coeffs['a'])  # Cross-sectional Area

P = 0.0  # Perimeter at position x
beta = 0.0  # Film coefficient
T_surr = 0.0  # Surrounding Temperature
c = -1. * P * beta
sdbc = 600.  # Start Dirchlet Boundary Conditions
edbc = 300.  # End Dirchlet Boundary Conditions
H = beta * T_surr
q = 0.  # Heat generated
F = 0.


# Analytical_Solution
X = sym.Symbol('X')
u_ana = np.array([[600.], [522.], [300.]])

# Shape_Functions

Lsf = Lagsf(X, NPE - 1)
sf = Lsf.f()
dsf = Lsf.df()

for k in range(n_ele):

    k_ele_s = np.zeros((NPE, NPE))
    f_ele_s = np.zeros((NPE, 1))
    k_ele_f = np.zeros((NPE, NPE))
    f_ele_f = np.zeros((NPE, 1))
    h = xn[k + (NPE - 1)] - xn[k]



        # body force function
    bf = -1 * (A * q + P * beta * T_surr)  # is equal to heat generated q plus convection effect

    # Elemental_Stiffness and Elemental Force Matrix
    # id=np.arange(0,19,1)

    # if (k < (n_ele - 2)):
    if (k < n_ele /3):


        for i in range(NPE):
            for j in range(NPE):
                ka=(0.05 + 0.000325 * sf[j])
                jac = 0.500000000000 * h
                # jac = sym.diff(x, X)  # defining jacobian
                func = ((ka * dsf[i] * dsf[j] )/ jac ** 2 )
                func_2 = bf * sf[i]
                Y=1.
                Gt = G_int.Integration(func, func_2, p, w, X,Y,jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness()
                k_ele_f[i][j] = float(k_ele_f[i][j] + I)
            I2 = Gt.gauss_quad_force()
            f_ele_f[i][0] = f_ele_f[i][0] + I2
    print(f_ele_f)
    print(k_ele_f)
    # if (k < (n_ele - 2)):
    if (k >= n_ele /3):

        for i in range(NPE):
            for j in range(NPE):
                kb=(0.04 + 0.000304 * sf[j])
                jac=0.500000000000 * h
                # jac = sym.diff(x, X)  # defining jacobian
                func = ((kb * dsf[i] * dsf[j]) / jac ** 2)
                func_2 = bf * sf[i]
                Y=1.
                Gt = G_int.Integration(func, func_2, p, w, X,Y,jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness()
                k_ele_s[i][j] = float(k_ele_s[i][j] + I)
            I2 = Gt.gauss_quad_force()
            f_ele_s[i][0] = f_ele_s[i][0] + I2


    print(f_ele_s)
    print(k_ele_s)
    # Global_Stiffness and Force Matrix
    for i in range(NPE):
        for j in range(NPE):
            k_glob[i + (NPE - 1) * k][j + (NPE - 1) * k] = k_glob[i + (NPE - 1) * k][j + (NPE - 1) * k] + k_ele_f[i][j] + k_ele_s[i][j]
        f_glob[i + (NPE - 1) * k] = f_glob[i + (NPE - 1) * k] + f_ele_f[i] + f_ele_s[i]

    # Dirichlet boundary conditions

    node_dbc = np.array([[1, (len(xn))], [sdbc,edbc]])
    for k in range(len(node_dbc[0])):
        for i in range(len(k_glob)):
            if node_dbc[0][k] - 1 == i:
                for j in range(len(k_glob[0])):
                    f_glob[j] = f_glob[j] - k_glob[j][int(node_dbc[0][k]) - 1] * node_dbc[1][k]
                    if i != j:
                        k_glob[i][j] = 0.0
                        k_glob[j][i] = 0.0
                    else:
                        k_glob[i][j] = 1.0
                    f_glob[i] = node_dbc[1][k]


# solving equations===================
print(f_glob)
print(k_glob)

solveobjec = res(sparse.csr_matrix(k_glob), f_glob)  # initialize the object
u = solveobjec.get_res()

print("Printing x coorinates of the given governing equation:\n", xn)
print("Printing FEA solution of the given governing equation:\n", np.around(u, decimals=2))


# # Error code===================
# norm = 0.
# error = np.zeros(len(xn))
# for i in range(len(xn)):
#     error[i] = abs(u[i] - u_ana[i])
#
# rms = sqrt(mean_squared_error(u_ana, u))
#
# # Error Printing===================
# print("Printing error of the given governing equation:\n", error)
# print("Printing error of the given governing equation:\n", rms)
#
# # finding L2 norm
# l2 = np.linalg.norm(error)
# print("The error in L2 norm is:", l2)
# Plotting Analytical vs Fea Solution


# # Creating 2-D grid of features
# [X, Y] = np.meshgrid(xn, u)
#
# fig, ax = plt.subplots(1, 1)
#
# Z = X ** 2 + Y ** 2
#
# # plots filled contour plot
# ax.contourf(X, Y, Z)
#
# ax.set_title('Filled Contour Plot')
# ax.set_xlabel('feature_x')
# ax.set_ylabel('feature_y')
#
# plt.show()
# plot_objec = plot_res(xn, u, u_ana)  # initialize the object
# plot_objec.get_plot()

