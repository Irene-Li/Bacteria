import time
import numpy as np
from StoEvolutionPS import *

epsilon = 0.1
a = 0.2
k = 1
u = 1e-5
phi_t = 0
phi_shift = 10

X = 128
dx = 1
T = 5e4
dt = 5e-3
n_batches = 100
initial_value = 0
flat = False

for phi_t in [-0.6, -0.7]:
	label = 'phi_t_{}_skewed_droplet'.format(phi_t)
	initial_value = phi_t

	start_time = time.time()
	solver = StoEvolutionPS(epsilon, a, k, u, phi_t, phi_shift)
	solver.initialise(X, dx, T, dt, n_batches, initial_value, flat=flat)
	solver.save_params(label)
	solver.print_params()
	solver.evolve(verbose=True)
	solver.save_phi(label)
	end_time = time.time()
	print('The simulation took: {}'.format(end_time - start_time))
