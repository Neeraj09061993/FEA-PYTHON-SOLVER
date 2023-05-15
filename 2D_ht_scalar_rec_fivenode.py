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
from Gauss_Integration import gauss_int  as G_int
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

filename = "mesh\square_mesh_2D.msh"
Mesh_obj = line(filename=filename)
Mesh_obj.readmshfile()
mesh=Mesh_obj.mesh
meshinfo=Mesh_obj.info
physical=Mesh_obj.physical
#"""Defining_Nodes"""
n_ele=meshinfo['Quads']
conn=mesh['Full_Connectivity_dim']
print(mesh['Coords'][0])

# n_ele=2


# n_coord=np.array([[1,1,0,0,1,1,1,2,2,1], [0,1,1,0,0.5,1,0,0,1,0.5]])
# print((n_coord)[0, :])
# xn=np.sort((n_coord)[:, 0])
#

# """Nodes per Element"""
NPE = len(conn[0])
NG = (len(conn[0])+2)

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



#Shape_Functions

Nf1=Y*0.25*(1+X)*(Y-1);
Nf2=Y*0.25*(1+X)*(Y+1);
Nf3=0.25*(1-X)*(Y+1);
Nf4=0.25*(1-X)*(1-Y);
Nf5=0.5*(1+X)*(1-Y**2);
Nf=np.array([Nf1,Nf2,Nf3,Nf4,Nf5]);

Ns1=Y*0.25*(1-X)*(Y+1);
Ns2=Y*0.25*(1-X)*(Y-1);
Ns3=0.25*(1+X)*(1-Y);
Ns4=0.25*(1+X)*(1+Y);
Ns5=0.5*(1-X)*(1-Y**2);
Ns=np.array([Ns1,Ns2,Ns3,Ns4,Ns5]);
# N1=0.25*X* Y**2 - 0.25 *X* Y + 0.25* Y**2 - 0.25 *Y;
# N2=0.25 *X *Y**2 + 0.25* X *Y + 0.25 *Y**2 + 0.25 *Y;
# N3=-0.25* X* Y - 0.25* X + 0.25* Y + 0.25;
# N4=0.25* X* Y - 0.25* X - 0.25* Y + 0.25;
# N5=-0.5 *X* Y**2 + 0.5* X - 0.5* Y**2 + 0.5;


# Lsf = Lagsf(X, NPE - 1)
# sf = Lsf.f()
# dsf = Lsf.df()
for k in range(n_ele):
    x=0.0
    y=0.0

# Elemental_Stiffness and Elemental Force Matrix
    if (k == 0):
        n_coord = mesh['Coords'][conn[k]]
        print(n_coord)
        xn = n_coord[:, 0]
        print(xn)
        yn = n_coord[:, 1]
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
        b = (2 - x ** 2 - y ** 2)

        for i in range(NPE):
            for j in range(NPE):
                jac = sym.diff(x, X)*sym.diff(y, Y)-sym.diff(x, Y)*sym.diff(y, X)  # defining jacobian
                print(jac)
                func = ((sym.diff(Nf[i], X) * sym.diff(Nf[j], X))+(sym.diff(Nf[i], Y) * sym.diff(Nf[j], Y)))/jac
                func_2 = Nf[i] * b
                Gt = G_int.Integration(func, func_2, p, w, X,Y, jac)  # Gauss Integration
                I = Gt.gauss_quad_stiffness_2D()
                k_ele_f[i][j] = float(k_ele_f[i][j] + I)
            I2 = Gt.gauss_quad_force_2D()
            f_ele_f[i] =(f_ele_f[i]+I2)

    if (k != 0):
        n_coord = mesh['Coords'][conn[k]]
        xn = n_coord[:, 0]
        # print(xn[1])
        yn = n_coord[:, 1]
        for i in range(NPE):
            # x_coor = float((n_coord)[0, i+5])
            x = x + Ns[i] * xn[i]
            # y_coor = float((n_coord)[1, i+5])
            y = y + Ns[i] * yn[i]
        k_ele_s = np.zeros((NPE, NPE))
        k_ele_f = np.zeros((NPE, NPE))
        f_ele_s = np.zeros((NPE, 1))
        f_ele_f = np.zeros((NPE, 1))
        m = (2 - x ** 2 - y ** 2)
        for i in range(NPE):
            for j in range(NPE):
                jac = sym.diff(x, X)*sym.diff(y, Y)-sym.diff(x, Y)*sym.diff(y, X)  # defining jacobian
                func = ((sym.diff(Ns[i], X) * sym.diff(Ns[j], X))+(sym.diff(Ns[i], Y) * sym.diff(Ns[j], Y)))/jac
                func_3 = Ns[i] * m
                Gt = G_int.Integration(func, func_3, p, w, X,Y, jac) # Gauss Integration
                I = Gt.gauss_quad_stiffness_2D()
                k_ele_s[i][j] = float(k_ele_s[i][j] + I)
            I2 = Gt.gauss_quad_force_2D()
            f_ele_s[i] = f_ele_s[i] + I2


    # Global_Stiffness and Force Matrix
    LM_1=np.array([1,5,4,0,3]);
    LM_2 = np.array([5, 1, 2, 6, 3]);


    for i in range(NPE):
        m = LM_1[i]
        r= LM_2[i]
        for j in range(NPE):
            n=LM_1[j]
            s = LM_2[j]
            k_glob[m][n] = k_glob[m][n] + k_ele_f[i][j]
            k_glob[r][s] = k_glob[r][s]+ k_ele_s[i][j]
        f_glob[m] = f_glob[m] + f_ele_f[i]
        f_glob[r] = f_glob[r] + f_ele_s[i]
    # print(f_glob)
    # print(k_glob)
 # Dirichlet boundary conditions

    node_dbc = np.array([[1,2,3,5,6,7], [0.,0.0,0.,0.,0.,0.]])

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
# solving equations     ===================
# print(k_glob)
# print(f_glob)
solveobjec = res(sparse.csr_matrix(k_glob), f_glob)  # initialize the object
u = solveobjec.get_res()
print("Printing Temperature at all nodes:\n", u)
u_r=0.
d={}
e={}
for i in range(5):
    u_r=u_r+Ns[i]*u[i]

d['X']=-1.
e['Y']=0
u_r=eval(str(u_r), d,e)
print("Printing Temperature at middle nodes of right side boundary:\n", u_r)
u_l=0.
g={}
h={}
for i in range(5):
    u_l=u_l+Ns[i]*u[i]

g['X']=-1.
h['Y']=0
u_l=eval(str(u_l), g,h)
print("Printing Temperature at middle nodes of left side boundary:\n", u_l)























