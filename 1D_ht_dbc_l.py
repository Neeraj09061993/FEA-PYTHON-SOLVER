import numpy as np
from math import sqrt
import sympy as sym
from sklearn.metrics import mean_squared_error

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

"""Calculates the element stiffness matrix and heat generated  contributions for a steady heat transfer problem,
        governed by the differential equation d/dx(EAd^u/dx)=B. Replace E=1., A=1.,B=-1. with appropriate
         quantities to solve  problems of the differential equation."""

# Find Gauss Points and Gauss Weight

NGP=3
G = GS.Gauss1d(NGP)



p, w = G.point_weight()
# print(w)
# print(p)

#=====Solve using Linear Finite Element ==============================================================================================

filename = "mesh/Line_mesh.msh"
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
d=1.0
"""Defining Coefficients of differential equations such as E value is 2*10^11 , A value is 30*10^-6 ,B value is 0 ,"""

coeff_obj = eaproperties(nel=n_ele, e=1., a=1., b=0.,bf=0.)
coeffs = coeff_obj.get_coeffs()
E=(coeffs['e'])
A=(coeffs['a'])
c=E*A
P=1.0 # Perimeter at position x
beta=0.0 #Film coefficient
q=1.0
T_surr=1.0

#Analytical_Solution
u_ana = np.zeros(len(xn))
for i in range(len(xn)):
	  # u_ana[i] =((((d*xn[i]**5.)/20.)+(100.*xn[i])-(d*xn[i]/4))/c)
	 # u_ana[i] = (100*xn[i])
     #u_ana[i] =(((xn[i]**5)/c*20.)+((99.75/c)*xn[i]))
     u_ana[i] =(-1.0*((xn[i]-1.0)*(xn[i]+200.0)/2.0))

#Shape_Functions

Lsf = Lagsf(X, NPE - 1)
sf = Lsf.f()
dsf = Lsf.df()

for k in range(n_ele):

	k_ele=np.zeros((NPE,NPE))
	f_ele = np.zeros((NPE, 1))

	x = 0.0

	for i in range(2):
		x_coor = float(n_coord[(conn[k,i]),0])
		x = x + sf[i] * x_coor
	b =  -1.


	#Elemental_Stiffness and Elemental Force Matrix

	for i in range(NPE):
		func_2 = (sf[i] * b)
		for j in range(NPE):
			jac = sym.diff(x, X) #defining jacobian
			func = ((c*-1 * sym.diff(sf[i], X) * sym.diff(sf[j], X)) / jac ** 2)
			Y = 1.
			Gt = G_int.Integration(func, func_2, p, w, X, Y, jac)  # Gauss Integration# Gauss Integration
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
	node_dbc = np.array([[1, len(xn)], [100.0, 0.0]])
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

solveobjec = res(sparse.csr_matrix(k_glob), f_glob)  # initialize the object
u = solveobjec.get_res()
# print(xn)
# print(k_glob)
# print(f_glob)
print("Printing x coorinates of the given governing equation:\n", xn)
print("Printing FEA solution of the given governing equation:\n", np.around(u, decimals = 2))
print("Printing Analytical solution  of the given governing equation:\n", np.around(u_ana, decimals = 2))
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






