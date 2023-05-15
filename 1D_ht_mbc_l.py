import numpy as np
from math import sqrt
import sympy as sym
from sklearn.metrics import mean_squared_error
from shapefunc.shp_fun import Lagsf
from mesh.readmsh import readmesh as line
from Solver.Solver import solution as res
from Elements.onedsecondorderode.coefficients import coefficients as eaproperties
from Plotter.Plot_heat import plotting as plot_res
from scipy import sparse
from gauss import gaussptwt as GS
from Gauss_Integration import gauss_int  as G_int


"""Calculates the element stiffness matrix and heat generated  contributions for a steady heat transfer problem,
        governed by the differential equation d/dx(KAd^u/dx)+Aq=B. Replace K,A,B. with appropriate
         quantities to solve  problems of the differential equation. Problem 4.3.1 JN Reddy book"""

# Find Gauss Points and Gauss Weight

NGP=3
G = GS.Gauss1d(NGP)


p, w = G.point_weight()
# print(w)
# print(p)

#=====Solve using Linear Finite Element ==============================================================================================

filename = "mesh\Line_4_elements_reddy.msh"
#filename = "mesh\Line_mesh.msh"
Mesh_obj = line(filename=filename)
Mesh_obj.readmshfile()
mesh=Mesh_obj.mesh
meshinfo=Mesh_obj.info
physical=Mesh_obj.physical
"""Defining_Nodes"""
conn=mesh['Connectivity']['Lines']
n_ele=meshinfo['Num_elem']
n_coord=mesh['Coords']
xn=np.sort((n_coord)[:, 0])


"""Nodes per Element"""
NPE=2

k_glob=np.zeros((len(xn),len(xn)))
f_glob=np.zeros((len(xn),1))


X=sym.Symbol('X')


coeff_obj = eaproperties(nel=n_ele, e=385000., a=5.e-6, b=1.)
coeffs = coeff_obj.get_coeffs()
ones = np.ones(n_ele)
K=(coeffs['e']) #Thermal Conductivity
A=(coeffs['a']) #Cross-sectional Area
P=0.012 # Perimeter at position x
beta=0.025 #Film coefficient
T_surr=20.0 #Surrounding Temperature
g=K*A
c=1.*P*beta
sdbc = 100. #Start Dirchlet Boundary Conditions
edbc=0. #End Dirchlet Boundary Conditions
H= A*beta*T_surr
q=0. #Heat generated

K= K*ones


#Analytical_Solution
X=sym.Symbol('X')
u_ana = np.array([[100.],[82.283],[70.732],[64.204],[62.053]])
# for i in range(len(xn)):
# 	#u_ana[i] = u_ana[i]-((5.*u_ana[i]+ 5.*(u_ana[i]-500.)*(sym.exp(2.*xn[i]))-101.*(sym.exp(xn[i]))*(sym.sinh(1.)) - 500.)*(sym.exp(-1*xn[i])))/(sym.sinh(1.))
#     u_ana[i] = u_ana[i]-(((1-101*sym.exp(1))*(sym.exp(xn[i])))//(1-sym.exp(2)))+101+((101-sym.exp(1))*(sym.exp(-1*xn[i])))/(2.*(sym.sinh(1.)))

#Shape_Functions

Lsf = Lagsf(X, NPE - 1)
sf = Lsf.f()
dsf = Lsf.df()

for k in range(n_ele):
    g=(K.item(k)*A)
    k_ele=np.zeros((NPE,NPE))
    f_ele = np.zeros((NPE, 1))

    x = 0.0

    for i in range(NPE):
        x_coor = float(n_coord[(conn[k,i]),0])
        x = x + sf[i] * x_coor # force matrix
        # body force function
    bf =(A*q+P*beta*T_surr) # is equal to heat generated q plus convection effect


	#Elemental_Stiffness and Elemental Force Matrix

    for i in range(NPE):
        
        func_2 = bf*sf[i]
        for j in range(NPE):
            jac = sym.diff(x, X) #defining jacobian

            func =((g*sym.diff(sf[i], X) * sym.diff(sf[j], X)) / jac ** 2+(c*sf[i] * sf[j]))
            Y=1.0
            Gt = G_int.Integration(func, func_2, p, w, X,Y, jac) # Gauss Integration
            I = Gt.gauss_quad_stiffness()
            k_ele[i][j] = float(k_ele[i][j] + I)
        I2 = Gt.gauss_quad_force()
        f_ele[i][0] = f_ele[i][0] + I2


	#Global_Stiffness and Force Matrix

    for i in range(NPE):
        for j in range(NPE):
            k_glob[i+k][j+k]= k_glob[i+k][j+k] + k_ele[i][j]
        f_glob[i + k] = f_glob[i + k] + f_ele[i]

	# Dirichlet boundary conditions
    node_dbc= np.array([[1],[sdbc]])
    for k in range(len(node_dbc[0])):
        for i in range(len(k_glob)):
            if node_dbc[0][k]-1==i:
                for j in range(len(k_glob[0])):
                    f_glob[j]= f_glob[j]- k_glob[j][int(node_dbc[0][k])-1]*node_dbc[1][k]
                    if i!=j:
                        k_glob[i][j]=0.0
                        k_glob[j][i]=0.0
                    else:
                        k_glob[i][j]= 1.0
                    f_glob[i]= node_dbc[1][k]
	# # Neumann boundary condition
    # print(f_glob)
    # node_nbc= np.array([[len(n_coord)],[H]])
    # for m in range(len(node_nbc[0])):
    #     for i in range(len(f_glob)):
    #         if node_nbc[0][m]-1 == i:
    #             f_glob[i] = f_glob[i]-node_nbc[1][m]

# solving equations===================
k_glob[-1][-1]=k_glob[-1][-1]+beta*A
f_glob[-1][-1]=f_glob[-1][-1]+beta*A*T_surr

solveobjec = res(sparse.csr_matrix(k_glob), f_glob)  # initialize the object
u = solveobjec.get_res()
Q_s=u[0]*k_ele[0][0]+u[1]*k_ele[0][1]-((P*beta*T_surr*xn[1])/2)
Q_l=(-beta*u[4]+beta*T_surr)

print("Printing x coorinates of the given governing equation:\n", xn)
print("Printing FEA solution of the given governing equation:\n", np.around(u, decimals = 2))
print("Printing Analytical solution  of the given governing equation:\n", u_ana)
print("Printing Heat Transfer value at end node of the given governing equation:\n", np.around(Q_s, decimals = 2))
print("Printing Heat Transfer value at end node of the given governing equation:\n", np.around(Q_l, decimals = 2))


# Plotting Analytical vs Fea Solution

plot_objec = plot_res(xn,u,u_ana)  # initialize the object
plot_objec.get_plot()

# Error code===================
norm = 0.
error = np.zeros(len(xn))
for i in range(len(xn)):
    error[i] = abs(u[i] - u_ana[i])

rms = sqrt(mean_squared_error(u_ana, u))

# Error Printing===================
print("Printing error of the given governing equation:\n", np.around(error, decimals = 4))
print("Printing rms error of the given governing equation:\n", np.around(rms, decimals = 4))

