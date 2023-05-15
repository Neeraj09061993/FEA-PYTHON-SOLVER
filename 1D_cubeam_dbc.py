import numpy as np
import sympy as sym
from sympy import *
# from scipy.special.orthogonal import p_roots
from shapefunc.shp_fun import Lagsf
from shapefunc.shp_beam import Lagbsf
from mesh.readmsh import readmesh as line
from Solver.Solver import solution as res
from Elements.onedsecondorderode.coefficients import coefficients as eaproperties
from Plotter.Plot import plotting as plot_res
from scipy import sparse
from gauss import gaussptwt as GS
from Gauss_Integration import gauss_int as G_int
from Elements.onedsecondorderode.secondorderode import odeelement as bar

"""Calculates the element stiffness matrix and body force contributions for a beam element,
        governed by the differential equation EI (d^4 u_y)/(dx^4 )  = p(x). Replace E,I with appropriate
         quantities to solve  problems of the differential equation."""

# Find Gauss Points and Gauss Weight

NGP=3;

G = GS.Gauss1d(NGP)

p, w = G.point_weight()


#=====Solve using Linear Finite Element =================
#   =============================================================================

# MESH DETAILS

filename = "mesh\Beam_problem_3.msh"
Mesh_obj = line(filename=filename)
Mesh_obj.readmshfile()
mesh=Mesh_obj.mesh
meshinfo=Mesh_obj.info
physical=Mesh_obj.physical

"""Defining_Nodes"""

conn=mesh['Connectivity']['Lines']
n_ele=meshinfo['Num_elem']
# print(n_ele)
n_coord=mesh['Coords']
xn=np.sort((n_coord)[:, 0])

"""Nodes per Element"""
NPE=4

k_glob=np.zeros((2*len(xn),2*len(xn)))
f_glob=np.zeros((2*len(xn),1))
X=symbols('X')

"""Defining Coefficients of differential equations such as E value is 1*10^4 , A value is 1. ,B value is 0 ,"""

coeff_obj = eaproperties(nel=n_ele, e=1*10**4., a=1., b=0.,bf=0.)
coeffs = coeff_obj.get_coeffs()
E=(coeffs['e'])
A=(coeffs['a'])


l_f = 8; # Length of First Element
l_s =4; # Length ofSecond Element
L = l_f+l_s; # Total Length
I= 1.; # Moment of Inertia of First Element

""" Different Constants """
c=E*I
P1 = -10.;
P2 = 5.;
P3 = -20.;
M1 = 20.;















""" Shape function for Jacobian calculation """

Lsf = Lagsf(X, 1)
sf = Lsf.f()

""" Elemental Stiffness Matrix ,Elemental Force Matrix """


for k in range(n_ele):
    k_ele_f = np.zeros((NPE, NPE))
    k_ele_s = np.zeros((NPE, NPE))
    f_ele_f = np.zeros((NPE, 1))

    # print(f_ele_f)
    f_ele_s = np.zeros((NPE, 1))

    x=0.0
    for i in range(2):
        x_coor = float(n_coord[(conn[k, i]), 0])
        x= x + sf[i] * x_coor

    b=-1.
    m=0.

    # Elemental_Stiffness and Elemental Force Matrix
    if (k == 0):
        for i in range(NPE):
            Lbsf = Lagbsf(X, l_f, 1)
            bsf = Lbsf.f()
            ddbsf = Lbsf.fx2()
            for j in range(NPE):
                jac = sym.diff(x, X)  # defining jacobian
                func = (((c * ddbsf[i]) * (ddbsf[j])) / jac ** 4)
                func_2 = bsf[i] * b
                Y = 1.
                Gt = G_int.Integration(func, func_2, p, w, X, Y, jac)  # Gauss Integration
                I= Gt.gauss_quad_stiffness()
                k_ele_f[i][j] = float(k_ele_f[i][j] + I)
            I2 = Gt.gauss_quad_force()
            Lbsf = Lagbsf(0, l_f, 1)
            bsf = Lbsf.f()
            f_ele_f[i] = (f_ele_f[i] + I2 + bsf[i] * P1)

    #print(k_ele_f)
    #print(f_ele_f)

    if (k != 0):
        for i in range(NPE):
            Lbsf = Lagbsf(X, l_s, 1)
            bsf = Lbsf.f()
            ddbsf = Lbsf.fx2()
            for j in range(NPE):
                jac = sym.diff(x, X)  # defining jacobian
                func = (((c * ddbsf[i]) * (ddbsf[j])) / jac ** 4)
                func_3 = bsf[i] * m
                Y = 1.
                Gt = G_int.Integration(func, func_3, p, w, X, Y, jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness()
                k_ele_s[i][j] = float(k_ele_s[i][j] + I)
            I2 = Gt.gauss_quad_force()
            Lbsf = Lagbsf(-1, l_s, 1)
            bsf = Lbsf.f()
            f_ele_s[i] = f_ele_s[i] + I2 + bsf[i] * P2

    #print(f_ele_s)
    # Global_Stiffness and Force Matrix

    for i in range(NPE):
        for j in range(4):
            k_glob[i + 2*k][j + 2*k] = k_glob[i + 2*k][j + 2*k] + k_ele_f[i][j]+k_ele_s[i][j]
        f_glob[i + 2*k] = f_glob[i + 2*k] + f_ele_f[i]+f_ele_s[i]
    # print(f_glob)
    # print(f_ele_s)



    # Dirichlet boundary conditions
    node_dbc = np.array([[1, 2], [0.0, 0.0]])

    for k in range(len(node_dbc[0])):
        for i in range(len(k_glob[0])):
            if node_dbc[0][k] - 1 == i:
                for j in range(len(k_glob[0])):
                    f_glob[j] = f_glob[j] - k_glob[j][int(node_dbc[0][k]) - 1] * node_dbc[1][k]
                    if i != j:
                        k_glob[i][j] = 0.0
                        k_glob[j][i] = 0.0
                    else:
                        k_glob[i][j] = 1.0
                f_glob[i] = node_dbc[1][k]

    # Neumann boundary condition

    node_nbc = np.array([[len(f_glob) - 1, len(f_glob)], [P3, M1]])
    for m in range(len(node_nbc[0])):
        for i in range(len(f_glob)):
            if node_nbc[0][m] - 1 == i:
                f_glob[i] = node_nbc[1][m]

# solving equations===================
# print(k_glob)
solveobjec = res(sparse.csr_matrix(k_glob), f_glob)  # initialize the object
u = solveobjec.get_res()
print("Deflections at nodes:")
print([u[0]],[u[2]],[u[4]])
print("Slopes at nodes:")
print([u[1]],[u[3]],[u[5]])
# # Deflection at 3 ft distance ===================
# print("Deflection at tip:")
# print(u[4])













