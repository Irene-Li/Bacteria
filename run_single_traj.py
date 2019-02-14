import time
import numpy as np
from StoEvolution import * 

M = 1e-6
a = 1e5 
k = 1e6
u = 2e-7
phi_t = 0
phi_shift = 100

X = 100
dx = 0.5
dt = 10
T = dt * 1e4
n_batches = 100
initial_value = 0 
flat = False

label = 'sto_u_{}'.format(u)

start_time = time.time()
solver = StoEvolution(M, a, k, u, phi_t, phi_shift)
solver.initialise(X, dx, T, dt, n_batches, initial_value, flat=flat)
solver.save_params(label)
solver.evolve(verbose=True)
solver.save_phi(label)
end_time = time.time()
print('The simulation took: {}'.format(end_time - start_time))