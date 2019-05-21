import time
import numpy as np
from StoEvolution import *

epsilon = 1e-4
a = 0.2
k = 1
u = 1e-5
phi_t = 0
phi_shift = 100

X = 40
dx = 0.2
T = 1000
n_batches = 100
initial_value = 0
flat = True

for dt in [1e-4]:
	label = 'sto_dt_{}_Long'.format(dt)

	start_time = time.time()
	solver = StoEvolution(epsilon, a, k, u, phi_t, phi_shift)
	solver.initialise(X, dx, T, dt, n_batches, initial_value, flat=flat)
	solver.save_params(label)
	solver._print_params()
	solver.evolve(verbose=True)
	solver.save_phi(label)
	end_time = time.time()
	print('The simulation took: {}'.format(end_time - start_time))


# evolve trajectories again with different parameters
# for dt in [100]:
# 	label = 'sto_dt_{}'.format(dt)
# 	print(label)
# 	solver.dt = dt
# 	solver.batch_size = int(solver.step_size/dt)
# 	solver.evolve_trajectories(n)
# 	solver.save_params(label)
# 	solver.save_trajs(label)
