import numpy as np
from FdEvolution import *

rates = [1e-8, 5e-8, 3e-7, 7e-7, 1e-7, 5e-7]


# parameters of the differential equation
a = 0.1
k = 1

phi_t = 0
phi_shift = 100

# simulation parameters
X = 200
dx = 0.1
dt = 1e-3
n_batches = 100

phi_shift = 100
phi_target = 0
phi_init = 0
random = False


for u in rates:
	label = 'X_200_u_{}'.format(u)
	print(label)
	T = 1/u
	solver = FdEvolution(a, k, u, phi_target, phi_shift)
	solver.initialise(X, dx, T, dt, n_batches, phi_init, random=random)
	solver.evolve()
	solver.save(label)
