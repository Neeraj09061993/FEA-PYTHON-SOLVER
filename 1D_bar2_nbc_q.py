import numpy as np
import sympy as sym
from shapefunc.shp_fun import Lagsf
from mesh.readmsh import readmesh as line
from Solver.Solver import solution as res
from Elements.onedsecondorderode.coefficients import coefficients as eaproperties
from Plotter.Plot_quad import plotting as plot_res
from scipy import sparse
from gauss import gaussptwt as GS
from Gauss_Integration import gauss_int  as G_int


"""Calculates the element stiffness matrix and body force contributions for a bar element,governed by the differential equation
-d/dx (du/dx)-u+x^2 = q(x).  Analytical solution : y(x)=x2−sin(x)/cos(1)+2cos(x−1)/cos(1)−2"""

# Find Gauss Points and Gauss Weight

NGP=3
G = GS.Gauss1d(NGP)


p, w = G.point_weight()
# print(w)
# print(p)

#=====Solve using Linear Finite Element ==============================================================================================

filename = "mesh\Line_mesh_three_quad.msh"
#filename = "mesh\Line_Mesh_1.msh"
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
NPE=3

k_glob=np.zeros((len(xn),len(xn)))
f_glob=np.zeros((len(xn),1))


X=sym.Symbol('X')

"""Defining Coefficients of differential equations such as E value is 2*10^11 , A value is 30*10^-6 ,B value is 0 ,"""

coeff_obj = eaproperties(nel=n_ele, e=1., a=1., b=0.)
coeffs = coeff_obj.get_coeffs()
E=(coeffs['e'])
A=(coeffs['a'])
g=E*A
c=-1.
F = 1.


#Analytical_Solution
X=sym.Symbol('X')
u_ana = np.zeros(len(xn))
for i in range(len(xn)):
	u_ana[i] = u_ana[i]+xn[i]**2.0 - ((sym.sin(xn[i]))/(sym.cos(1))) +(2*(sym.cos(xn[i]-1)))/(sym.cos(1)) - 2
	# print(u_ana)

#Shape_Functions

Lsf = Lagsf(X, NPE - 1)
sf = Lsf.f()
dsf = Lsf.df()

for k in range(n_ele):

	k_ele=np.zeros((NPE,NPE))
	f_ele = np.zeros((NPE, 1))

	x = 0.0

	for i in range(NPE):
		x_coor = float(n_coord[(conn[k,i]),0])
		x = x + sf[i] * x_coor
		# force matrix
		# body force function
		f = -x ** 2


	#Elemental_Stiffness and Elemental Force Matrix

	for i in range(NPE):
		func_2 = (sf[i] * f)
		for j in range(NPE):
			jac = sym.diff(x, X) #defining jacobian

			func = ((g*sym.diff(sf[i], X) * sym.diff(sf[j], X)) / jac ** 2-(sf[i] * sf[j]))
			Gt = G_int.Integration_1(func, func_2, p, w, X, jac) # Gauss Integration
			I = Gt.gauss_quad_stiffness()
			k_ele[i][j] = float(k_ele[i][j] + I)
		I2 = Gt.gauss_quad_force()
		f_ele[i][0] = f_ele[i][0] + I2


	#Global_Stiffness and Force Matrix

	for i in range(NPE):
		for j in range(NPE):
			k_glob[i+2*k][j+2*k]= k_glob[i+2*k][j+2*k] + k_ele[i][j]
		f_glob[i + 2*k] = f_glob[i + 2*k] + f_ele[i]


	# Dirichlet boundary conditions
	node_dbc= np.array([[1],[0.0]])
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
	# Neumann boundary condition
	
	node_nbc= np.array([[len(n_coord)],[F]])
	for m in range(len(node_nbc[0])):
		for i in range(len(f_glob)):
			if node_nbc[0][m]-1 == i:
				f_glob[i] =  node_nbc[1][m]

# solving equations===================

solveobjec = res(sparse.csr_matrix(k_glob), f_glob)  # initialize the object
u = solveobjec.get_res()
print("Printing x coorinates of the given governing equation:\n", xn)
print("Printing FEA solution of the given governing equation:\n", u)
print("Printing Analytical solution  of the given governing equation:\n", u_ana)
# Plotting Analytical vs Fea Solution

plot_objec = plot_res(xn,u,u_ana)  # initialize the object
plot_objec.get_plot()





