import numpy as np
import sympy as sym
# import only system from os
from os import system, name
  
# import sleep to show output for some time period
from time import sleep

# from scipy.special.orthogonal import p_roots
from shapefunc.shp_fun import Lagsf
from mesh.readmsh import readmesh as line
from Solver.Solver import solution as res
from Elements.onedsecondorderode.coefficients import coefficients as eaproperties
from Plotter.Plot_quad import plotting as plot_res
from scipy import sparse
from gauss import gaussptwt as GS
from Gauss_Integration import gauss_int  as G_int
from Elements.onedsecondorderode.secondorderode import odeelement as bar

"""Calculates the element stiffness matrix and body force contributions for a bar element,
        governed by the differential equation d/dx(EAd^u/dx)=B. Replace E=1., A=1.,B=0. with appropriate
         quantities to solve  problems of the differential equation."""

# Find Gauss Points and Gauss Weight

NGP=3
G = GS.Gauss1d(NGP)


p, w = G.point_weight()
# print(w)
# print(p)

#=====Solve using Linear Finite Element ==============================================================================================

filename = "mesh\Line_mesh_three_quad.msh"
Mesh_obj = line(filename=filename)
Mesh_obj.readmshfile()
mesh=Mesh_obj.mesh
meshinfo=Mesh_obj.info
physical=Mesh_obj.physical
"""Defining_Nodes"""
conn=mesh['Connectivity']['Lines']
print(conn)
ele_conn=np.sort((conn))

n_ele=meshinfo['Num_elem']
n_coord=mesh['Coords']
xn=np.sort((n_coord)[:, 0])
xne=((n_coord)[:, 0])
#print(xne)

"""Nodes per Element"""
NPE=3

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

#Analytical_Solution
u_ana = np.zeros(len(xn))
for i in range(len(xn)):
	#u_ana[i] = u_ana[i]+xn[i]**2.0 - (sym.sin(xn[i]-1))/(sym.sin(2.0)) + (2*sym.sin(xn[i]+1))/(sym.sin(2)) - 2
 	u_ana[i] = (100*xn[i])

#Shape_Functions

Lsf = Lagsf(X, NPE - 1)
sf = Lsf.f()

dsf = Lsf.df()

ele_con= np.zeros([n_ele,4])
for i in range(n_ele):
    ele_con[i]= [i, 2*i+1, 2*i+2, 2*i+3]


for k in range(n_ele):

	k_ele=np.zeros((NPE,NPE))
	f_ele = np.zeros((NPE, 1))

	x = 0.0

	for i in range(NPE):
		x_coor = float(n_coord[(conn[k,i]),0])
		x = x + sf[i] * x_coor
	b = 0.


	#Elemental_Stiffness and Elemental Force Matrix

	for i in range(NPE):
		func_2 = (sf[i] * b)
		for j in range(NPE):
			jac = sym.diff(x, X) #defining jacobian
			func = (c * dsf[i] * dsf[j] / jac ** 2)
			Gt = G_int.Integration_1(func, func_2, p, w, X, jac) # Gauss Integration
			I = Gt.gauss_quad_stiffness()
			k_ele[i][j] = float(k_ele[i][j] + I)
		
		I2 = Gt.gauss_quad_force()
		f_ele[i][0] = f_ele[i][0] + I2


	#Global_Stiffness and Force Matrix

	for i in range(NPE):
		for j in range(NPE):
			k_glob[i+2*k][j+2*k]= k_glob[i+2*k][j+2*k] + k_ele[i][j]
		f_glob[i+2*k]= f_glob[i+2*k] + f_ele[i]

	# Dirichlet boundary conditions
	node_dbc = np.array([[1, len(xn)], [0., 100.0]])
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
print("Printing x coorinates of the given governing equation:\n", xn)
print("Printing FEA solution of the given governing equation:\n", u)
print("Printing Analytical solution  of the given governing equation:\n", u_ana)
# Plotting Analytical vs Fea Solution

plot_objec = plot_res(xn,u,u_ana)  # initialize the object
plot_objec.get_plot()
  






