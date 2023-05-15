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

"""Defining Coefficients of differential equations such as E value is 2*10^7 , A value is 30*10^-6 ,B value is 0 ,"""

coeff_obj = eaproperties(nel=n_ele, e=2*10**7., a=1., b=0.,bf=0.)
coeffs = coeff_obj.get_coeffs()
E=(coeffs['e'])
A=(coeffs['a'])


l_f = 8; # Length of First Element
l_s =6; # Length ofSecond Element
L = l_f+l_s; # Total Length
I= 1.; # Moment of Inertia of First Element

""" Different Constants """

c=E*I

P1=0.
P2 = 2000.;


"""Start Global Shape_Functions for Beam Element"""

# N1=1-3*(X/h)**2+2*(X/h)**3;
# N2=-X*(1-X/h)**2;
# N3=3*(X/h)**2-2*(X/h)**3;
# N4=-X*((X/h)**2-(X/h));

""" End Global Shape_Functions for Beam Element"""

"""Local Shape_Functions for First Beam Element"""

# Nf1 = .25 * (1 - X) **2 * (2 + X);
# Nf2 = l_f * .125 * (1 - X)**2* (1 + X);
# Nf3 = .25 * (1 + X) **2 * (2 - X);
# Nf4 = l_f * .125 * (1 + X) **2* (X - 1);
# Nf=np.array([Nf1 , Nf2 , Nf3 , Nf4]);

"""Local Shape_Functions for First Beam Element @ specific values of X=0.33"""
#
# Nf_s1 = .25 * (1 - .33) **2 * (2 + .33);
# Nf_s2 = l_f * .125 * (1 - .33)**2* (1 + .33);
# Nf_s3 = .25 * (1 + .3) **2 * (2 - .33);
# Nf_s4 = l_f * .125 * (1 + .33) **2* (.33 - 1);
# Nf_s=np.array([[Nf_s1],[Nf_s2],[Nf_s3],[Nf_s4]]);


"""Local Shape_Functions for Second Beam Element """

# Ns1 = .25 * (1 - X) **2 * (2 + X);
# Ns2 = l_s * .125 * (1 - X)**2* (1 + X);
# Ns3 = .25 * (1 + X) **2 * (2 - X);
# Ns4 = l_s * .125 * (1 + X) **2* (X - 1);
# Ns = np.array([Ns1 , Ns2 , Ns3 , Ns4]);

"""Local Shape_Functions for Second Beam Element @ specific values of X=1"""
# Ns_s1 = .25 * (1 - 1) **2 * (2 + 1);
# Ns_s2 = l_s * .125 * (1 - 1)**2* (1 + 1);
# Ns_s3 = .25 * (1 + 1) **2 * (2 - 1);
# Ns_s4 = l_s * .125 * (1 + 1) **2* (1- 1);
# Ns_s=np.array([Ns_s1,Ns_s2,Ns_s3,Ns_s4]);

"""Second Order Differentiation of Local Shape_Functions for First Beam Element"""

# dNf1=(sym.diff(sym.diff(Nf1,X)));
# dNf2=(sym.diff(sym.diff(Nf2,X)));
# dNf3=(sym.diff(sym.diff(Nf3,X)));
# dNf4=(sym.diff(sym.diff(Nf4,X)));
# dNf=np.array([dNf1,dNf2,dNf3,dNf4]);

"""Second Order Differentiation of Local Shape Functions for Second Beam Element"""

# dNs1=(sym.diff(sym.diff(Ns1,X)));
# dNs2=(sym.diff(sym.diff(Ns2,X)));
# dNs3=(sym.diff(sym.diff(Ns3,X)));
# dNs4=(sym.diff(sym.diff(Ns4,X)));
# dNs=np.array([dNs1,dNs2,dNs3,dNs4]);

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
    r=1000/(l_f*l_f)
    b=1000-r*16*(1+X)*(1+X)
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
            Lbsf = Lagbsf(0.33, l_f, 1)
            bsf = Lbsf.f()
            f_ele_f[i] = (f_ele_f[i] + I2 + bsf[i] * P1)



    if (k != 0):
        for i in range(NPE):
            Lbsf = Lagbsf(X, l_s, 1)
            bsf = Lbsf.f()
            ddbsf = Lbsf.fx2()
            for j in range(NPE):
                jac = sym.diff(x, X)  # defining jacobian
                func = (((c*ddbsf[i]) * (ddbsf[j])) / jac ** 4)
                func_3 = bsf[i] * m
                Y = 1.
                Gt = G_int.Integration(func, func_3, p, w, X,Y, jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness()
                k_ele_s[i][j] = float(k_ele_s[i][j] + I)
            I2 = Gt.gauss_quad_force()
            Lbsf = Lagbsf(1, l_s, 1)
            bsf = Lbsf.f()
            f_ele_s[i] = f_ele_s[i] + I2 + bsf[i] * P2


    # Global_Stiffness and Force Matrix

    for i in range(NPE):
        for j in range(4):
            k_glob[i + 2*k][j + 2*k] = k_glob[i + 2*k][j + 2*k] + k_ele_f[i][j]+k_ele_s[i][j]
        f_glob[i + 2*k] = f_glob[i + 2*k] + f_ele_f[i]+f_ele_s[i]




    # Dirichlet boundary conditions
    node_dbc = np.array([[1, 2], [0., 0.0]])

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



# solving equations===================
# print(f_glob)
solveobjec = res(sparse.csr_matrix(k_glob), f_glob)  # initialize the object
u = solveobjec.get_res()
print("Deflections at nodes:")
print([u[0]],[u[2]],[u[4]])
print("Slopes at nodes:")
print([u[1]],[u[3]],[u[5]])
# Deflection at 3 ft distance ===================
print("Deflection at tip:")
print(u[4])













