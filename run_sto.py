import time
import numpy as np
from StoEvolution import *

epsilon = 1e-4
a = 0.2
k = 1
u = 0
phi_t = -0.3
phi_shift = 1.05

X = 50
dx = 0.1
dt = 1e-3
T = dt * 5e2
n_batches = 50
initial_value = -0.6
flat = True
# n = 5

label = 'sto_dt_{}'.format(u)

start_time = time.time()
solver = StoEvolution(epsilon, a, k, u, phi_t, phi_shift)
solver.initialise(X, dx, T, dt, n_batches, initial_value, flat=flat)
solver.save_params(label)
solver._print_params()
solver.evolve(verbose=True)
solver.save_phi(label)
# solver.evolve_trajectories(n)
# solver.save_trajs(label)
end_time = time.time()
print('The simulation took: {}'.format(end_time - start_time))
print('Time per trajectory: {}'.format((end_time - start_time)/n))


# evolve trajectories again with different parameters
# for dt in [100]:
# 	label = 'sto_dt_{}'.format(dt)
# 	print(label)
# 	solver.dt = dt
# 	solver.batch_size = int(solver.step_size/dt)
# 	solver.evolve_trajectories(n)
# 	solver.save_params(label)
# 	solver.save_trajs(label)
