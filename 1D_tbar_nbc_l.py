import numpy as np
import sympy as sym
# from scipy.special.orthogonal import p_roots
from shapefunc.shp_fun import Lagsf
from mesh.readmsh import readmesh as line
from Solver.Solver import solution as res
from Elements.onedsecondorderode.coefficients import coefficients as eaproperties
from Plotter.Plot import plotting as plot_res
from scipy import sparse
from gauss import gaussptwt as GS
from Gauss_Integration import gauss_int as G_int



"""Calculates the element stiffness matrix and body force contributions for a bar element,
        governed by the differential equation d/dx(EAd^u/dx)=B. Replace E=1., A=1.,B=0. with appropriate
         quantities to solve  problems of the differential equation. J N Reddy Example 4.5.2 Page No.188 """

# Find Gauss Points and Gauss Weight

NGP = 3
G = GS.Gauss1d(NGP)

p, w = G.point_weight()
# print(p)
# print(w)

# =====Solve using Linear Finite Element =================
#   =============================================================================

filename = "mesh\Tapered_bar_problem.msh"
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

"""Nodes per Element"""
NPE = 2

k_glob = np.zeros((len(xn), len(xn)))
f_glob = np.zeros((len(xn), 1))

X = sym.Symbol('X')
Y = sym.Symbol('Y')
d = 1.0
"""Defining Coefficients of differential equations such as E value is 2*10^11 , A value is 30*10^-6 ,B value is 0 ,"""

coeff_obj = eaproperties(nel=n_ele, e=2 * 10 ** 11., a=30 * 10 ** -6., b=0., bf=0.)
coeffs = coeff_obj.get_coeffs()
E = (coeffs['e'])
A = (coeffs['a'])
#Aluminium
E_a = 1 * 10 ** 7  # in psi.
A_a= 1 # in^2

c_a=E_a*A_a
#Steel

E_s = 30 * 10 ** 6  # in psi.
l_s = 96 # in
c1 = 1.5
c2 = (A_a-c1)/l_s
P0 = 10000
P1 = 2*P0
R1=0


# Shape_Functions

Lsf = Lagsf(X, NPE - 1)
sf = Lsf.f()
dsf = Lsf.df()

for k in range(n_ele):


    k_ele_s = np.zeros((NPE, NPE))

    f_ele_s = np.zeros((NPE, 1))

    x = 0.0

    for i in range(2):
        x_coor = float(n_coord[(conn[k, i]), 0])

        x = x + sf[i] * x_coor
        Y = sf[i] * x_coor
    b = 0
    m = 0


    # Elemental_Stiffness and Elemental Force Matrix
    if (k == 0):
        k_ele_f = np.zeros((NPE, NPE))
        f_ele_f = np.zeros((NPE, 1))
        for i in range(NPE):
            for j in range(NPE):
                jac = sym.diff(x, X)  # defining jacobian

                func = ((E_s * (c1+c2*Y)**2 * dsf[i] * dsf[j]) / jac ** 2)
                func_2 = sf[i] * b
                Y = 1.
                Gt = G_int.Integration(func, func_2, p, w, X, Y, jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness()
                k_ele_f[i][j] = float(k_ele_f[i][j] + I)
            I2 = Gt.gauss_quad_force()
            f_ele_f[i] =(f_ele_f[i]+I2)
    # print(k_ele_f)
    if (k != 0):
        for i in range(NPE):
            for j in range(NPE):
                jac = sym.diff(x, X)  # defining jacobian
                # func = (((c_s * dNs[i]) * (dNs[j])) / jac ** 4)
                func = ((c_a * dsf[i] * dsf[j]) / jac ** 2)
                func_3 = sf[i] * m
                Y = 1.
                Gt = G_int.Integration(func, func_2, p, w, X, Y, jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness()
                k_ele_s[i][j] = float(k_ele_s[i][j] + I)
            I2 = Gt.gauss_quad_force()
            f_ele_s[i] = f_ele_s[i] + I2

    # print(k_ele_s)
    # Global_Stiffness and Force Matrix

    for i in range(NPE):
        for j in range(NPE):
            k_glob[i + k][j + k] = k_glob[i + k][j + k] + k_ele_f[i][j]+k_ele_s[i][j]
        f_glob[i + k] = f_glob[i + k] + f_ele_f[i]+f_ele_s[i]

    # Dirichlet boundary conditions
    node_dbc = np.array([[1], [0.]])
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

    node_nbc = np.array([[(len(f_glob)-1), (len(f_glob))], [P1,P0]])
    for m in range(len(node_nbc[0])):
        for i in range(len(f_glob)):
            if node_nbc[0][m] - 1 == i:
                f_glob[i] = node_nbc[1][m]
# solving equations===================
# print(f_glob)
# print(k_glob)

solveobjec = res(sparse.csr_matrix(k_glob), f_glob)  # initialize the object
u = solveobjec.get_res()
for i in range(len(f_glob)-1):
    R1 = R1+u[i]*k_ele_f[0][i]


# Printing solutions===================
print("Printing x coordinates of the given governing equation:\n", xn)
print("Printing FEA solution of the given governing equation:\n", u)
print("Printing Reaction Force at left end:\n", R1)







