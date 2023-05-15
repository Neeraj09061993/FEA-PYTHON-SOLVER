import numpy as np
import sympy as sym
from shapefunc.shp_fun import Lagsf
from mesh.readmsh import readmesh as line
from Elements.onedsecondorderode.coefficients import coefficients as eaproperties
from gauss import gaussptwt as GS
from Gauss_Integration import gauss_int_mass as G_int


"""Calculates the eigen values , eigen vectors and eigen functions for a element
        governed by the differential equation d/dx(kd^T/dx)=rho*d*dT/dt. Replace E=1., A=1.,B=0. with appropriate
         quantities to solve  problems of the differential equation.J.N. Reddy's example 6.1.1 for set 2 boundary conditions. """

# Find Gauss Points and Gauss Weight

NGP = 3
G = GS.Gauss1d(NGP)
p, w = G.point_weight()

# =====Solve using Linear Finite Element ==============================================================================================

filename = "mesh/Line_2_elements.msh"
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
NPE = n_ele

"""Dimensions of global matrices"""

k_glob = np.zeros((len(xn), len(xn)))
k_new = np.zeros((len(xn), len(xn)))
m_glob = np.zeros((len(xn), len(xn)))
f_glob = np.zeros((len(xn), 1))

"""Different variables and constants defined"""

X = sym.Symbol('X')
L = sym.Symbol('L')
d = 1.0
rho = 1.0

"""Defining Coefficients of differential equations such as E value is 2*10^11 , A value is 30*10^-6 ,B value is 0 ,"""

coeff_obj = eaproperties(nel=n_ele, e=1., a=1., b=0., bf=0.)
coeffs = coeff_obj.get_coeffs()
E = (coeffs['e'])
A = (coeffs['a'])
c = E * A

# Shape_Functions

Lsf = Lagsf(X, NPE - 1)
sf = Lsf.f()
dsf = Lsf.df()

for k in range(n_ele):

    k_ele = np.zeros((NPE, NPE))
    m_ele = np.zeros((NPE, NPE))
    f_ele = np.zeros((NPE, 1))

    x = 0.0  # initial x coordinates

    for i in range(2):
        x_coor = float(n_coord[(conn[k, i]), 0])
        x = x + sf[i] * x_coor
    b = 0.  # body force constant

    # Elemental_Stiffness and Elemental Force Matrix

    for i in range(NPE):
        force_func = (sf[i] * b)
        for j in range(NPE):
            jac = sym.diff(x, X)  # defining jacobian
            stiff_func = ((c *dsf[i] * dsf[j]) / jac ** 2)
            mass_func = (rho * A * sf[i] * sf[j])
            Gt = G_int.Integration(stiff_func, mass_func, force_func, p, w, X, jac)  # Gauss Integration
            I = Gt.gauss_quad_stiffness()
            I3 = Gt.gauss_quad_mass()
            k_ele[i][j] = float(k_ele[i][j] + I)
            m_ele[i][j] = float(m_ele[i][j] + I3)
        I2 = Gt.gauss_quad_force()
        f_ele[i][0] = f_ele[i][0] + I2

    # Global_Stiffness and Force Matrix

    for i in range(NPE):
        for j in range(NPE):
            k_glob[i + k][j + k] = k_glob[i + k][j + k] + k_ele[i][j]
            m_glob[i + k][j + k] = m_glob[i + k][j + k] + m_ele[i][j]
        f_glob[i + k] = f_glob[i + k] + f_ele[i]

    # Dirichlet boundary conditions
    node_dbc = np.array([[1], [0.]])
    for k in range(len(node_dbc[0])):
        for i in range(len(k_glob[0])):
            if node_dbc[0][k] - 1 == i:
                for j in range(len(k_glob[0])):
                    f_glob[j] = f_glob[j] - k_glob[j][int(node_dbc[0][k]) - 1] * node_dbc[1][k] - m_glob[j][
                        int(node_dbc[0][k]) - 1] * node_dbc[1][k]
                    if i != j:
                        k_glob[i][j] = 0.0
                        k_glob[j][i] = 0.0
                        m_glob[i][j] = 0.0
                        m_glob[j][i] = 0.0

                    else:
                        k_glob[i][j] = 1.0
                        m_glob[i][j] = 1.0
                f_glob[i] = node_dbc[1][k]

# finding inverse of mass matrix and new global stiffness matrix
H = 1.
inv_m = np.linalg.inv(m_glob)
k_glob[-1][-1]=k_glob[-1][-1]+H
k_new = np.dot(k_glob, inv_m)

# finding eigenvalues and eigenvectors

w, v = np.linalg.eig(k_new)

# printing eigen values
print("Printing the Eigen values of the given square array:\n", w)

# printing eigen vectors
print("Printing Right eigenvectors of the given square array:\n", v)























