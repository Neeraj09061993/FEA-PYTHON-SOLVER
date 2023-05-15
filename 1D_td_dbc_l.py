import numpy as np
import sympy as sym
# from scipy.special.orthogonal import p_roots
from shapefunc.shp_fun import Lagsf
from sklearn.metrics import mean_squared_error
from math import sqrt
from mesh.readmsh import readmesh as line
from Solver.Solver import solution as res
from Elements.onedsecondorderode.coefficients import coefficients as eaproperties
# =============================================
from conditions.definebc import definebcs
from conditions.applybc import bcapply
from conditions.initialconditions import initial_one_d as initial
# =============================================
from Plotter.Plot import plotting as plot_res
from scipy import sparse
from gauss import gaussptwt as GS
from Gauss_Integration import gauss_int_mass as G_int
from output.outputview import output_point_data as point_data
import matplotlib.pyplot as plt
from Elements.onedsecondorderode.secondorderode import odeelement as bar

"""Calculates the element stiffness matrix and body force contributions for a bar element,
        governed by the differential equation d/dx(EAd^u/dx)=B. Replace E=1., A=1.,B=0. with appropriate
         quantities to solve  problems of the differential equation."""

# Find Gauss Points and Gauss Weight

NGP = 2
G = GS.Gauss1d(NGP)
p, w = G.point_weight()
# print("Printing gauss points:\n", p)
# print("Printing gauss weights:\n", w)

# =====Solve using Linear Finite Element ==============================================================================================

filename = "mesh/Line_5_element.msh"
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
k_new = np.zeros((len(xn), len(xn)))
m_glob = np.zeros((len(xn), len(xn)))
k_alpha = np.zeros((len(xn), len(xn)))
m_alpha = np.zeros((len(xn), len(xn)))
m_alpha_inc = np.zeros((len(xn), len(xn)))
f_glob = np.zeros((len(xn), 1))





# ===================================================
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
    h = xn[k + 1] - xn[k]


    b = 0.

    # Elemental_Stiffness and Elemental Force Matrix

    for i in range(NPE):
        force_func = (sf[i] * b)
        for j in range(NPE):
            jac=0.500000000000 * h # defining jacobian
            stiff_func = ((c * sym.diff(sf[i], X) * sym.diff(sf[j], X)) / jac ** 2)
            mass_func = (rho * A * sf[i] * sf[j])

            Gt = G_int.Integration(stiff_func, mass_func, force_func, p, w, X, jac)  # Gauss Integration
            I = Gt.gauss_quad_stiffness()
            I3 = Gt.gauss_quad_mass()

            k_ele[i][j] = float(k_ele[i][j] + I)

            m_ele[i][j] = float(m_ele[i][j] + I3)

        I2 = Gt.gauss_quad_force()
        f_ele[i][0] = f_ele[i][0] + I2
    #print(k_ele)
    #print(f_ele)
    # print(m_ele)
    # Global_Stiffness and Force Matrix

    for i in range(NPE):
        for j in range(NPE):
            k_glob[i + k][j + k] = k_glob[i + k][j + k] + k_ele[i][j]
            m_glob[i + k][j + k] = m_glob[i + k][j + k] + m_ele[i][j]

        f_glob[i + k] = f_glob[i + k] + f_ele[i]
    #print(k_glob)
    #print(m_glob)
    # print(f_glob)
# Dirichlet boundary conditions
    node_dbc = np.array([[1, len(xn)], [0.0, 0.0]])
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


# print(k_glob)
# print(m_glob)
# print(f_glob)
# Conditional Stability

inv_m = np.linalg.inv(m_glob)
k_new = np.dot(k_glob, inv_m)

# finding eigenvalues and eigenvectors
w, v = np.linalg.eig(k_new)

# printing eigen values
# print("Printing the Eigen values of the given square array:\n", w)

alpha = 0

delta_t_cri = 2 / w[0]

b=0.
# Define parameters needed for time-stepping.
t_final = 0.2
for n in range(1):

    # delta_t = delta_t_cri /(n+1)
    delta_t =0.1
    a1 = alpha * delta_t
    a2 = (1 - alpha) * delta_t
    nsteps = int(t_final / delta_t)+1

    ic = np.zeros(len(xn))
    for i in range(len(xn)):
        ic[i] = 1 - (xn[i] ** 2)
    uprev = ic
    time = 0

    t = np.zeros(nsteps)

    x = np.zeros(nsteps)
    up = np.zeros((nsteps+1, len(xn)))
    up[0,:]=uprev

    u_ana = np.array([[0 ,.64 ,.96 ,.96 ,.64 , 0],[0 ,.38 ,.77 ,.77 ,.38 ,0],[0 ,.46 ,.47 ,.47 ,.46 ,0]])
    error = np.zeros(len(xn))
    for i in range(nsteps):
        time = delta_t * (i)
        print("NSTEPS:Time", i, time)
        K_bar_s = m_glob - a2 * k_glob
        K_hat_sp1 = m_glob + a1 * k_glob

        F_bar = delta_t * (alpha * f_glob + (1 - alpha) * f_glob)

        F_s = f_glob
        RHS = sparse.csr_matrix(K_bar_s).dot(np.transpose(uprev)) + np.transpose(F_bar)

        solveobjec = res(sparse.csr_matrix(K_hat_sp1), np.transpose(RHS))
        u = solveobjec.get_res()
        t[i] = i * delta_t
        x[i] = u[1]
        u_p = []
        for j in range(len(xn)):
            up[i+1][j] = up[i+1][j] + u[j]
            u_p.append(up[0:3, j])
        uprev = u
    # print(xn)
    # print(t)
    # print(up)



    # # # Data for a two-dimensional line
    plt.figure()  # In this example, all the plots will be in one figure.
    for i in range(len(xn)):

        # plt.plot(t, up[:, i], label='T @ x_coord= %.2f ' % (xn[i]))
        plt.plot(t, up[0:3, i], label='T_FEA @ x_coord= %.2f ' % (xn[i]))
        plt.plot(t,u_ana[:,i],  'o', label='T_ANA')
        # plt.plot(t,u_ana[:,i],'-')
    plt.legend(loc='upper right')
    plt.xlabel('Time coordinates')
    plt.ylabel('Temperature')
    plt.title('Time vs Temperature for  5 linear elements')
    plt.grid(True)
    plt.show()

    # # # Data for a two-dimensional line
    plt.figure()  # In this example, all the plots will be in one figure.
    for i in range(nsteps):
        rms = sqrt(mean_squared_error(u_ana[ i , :], up[i, 0:6]))
        print("Printing error of the given governing equation:\n",np.around(rms, decimals=4))
        plt.plot(xn, up[i, 0:6], label='T_FEA @ t_step = %.f  ' % (i))
        plt.plot(xn, u_ana[i, :], 'o',label='T_ANA @ t_step = %.f  ' % (i))
    plt.legend(loc='upper right')
    plt.xlabel('Space coordinates')
    plt.ylabel('Temperature')
    plt.title('Space vs Temperature for  5 linear elements')
    plt.grid(True)
    plt.show()

    # Data for a three-dimensional line

    # # Creating 2-D grid of features
    [X, Y] = np.meshgrid(t, xn)
    fig, ax = plt.subplots(1, 1)
    # plots filled contour plot

    ax.contourf(t, xn, u_p)
    ax.set_title('Temperature Contour Plot', size=12)
    ax.set_xlabel('Time coordinates', size=12)
    ax.set_ylabel('Space coordinates', size=12)
    plt.show()

































