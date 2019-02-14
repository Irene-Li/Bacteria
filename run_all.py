import numpy as np
from FdEvolution import *

rates = [6e-6, 7e-6, 8e-6, 9e-6, 1.1e-5]


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
	T = 0.5/u
	solver = FdEvolution(a, k, u, phi_target, phi_shift)
	solver.initialise(X, dx, T, dt, n_batches, phi_init, random=random)
	solver.evolve()
	solver.save(label)
