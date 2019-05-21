import numpy as np
from FdEvolution import *

rates = [3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]


# parameters of the differential equation
a = 0.1
k = 1

phi_t = 0
phi_shift = 100

# simulation parameters
X = 500
dx = 0.1
dt = 1e-3
n_batches = 100

phi_shift = 100
phi_target = 0
phi_init = 0


for u in rates:
	for init in ['flat', 'sin']:
		label = '{}_X_500_u_{}'.format(init, u)
		random = (init == 'flat')
		print(label)
		T = 5/u
		solver = FdEvolution(a, k, u, phi_target, phi_shift)
		solver.initialise(X, dx, T, dt, n_batches, phi_init, random=random)
		solver.evolve()
		solver.save(label)
