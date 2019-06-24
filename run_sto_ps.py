import time
import numpy as np
from StoEvolutionPS import *

epsilon = 0.01
a = 0.2
k = 1
phi_t = -0.65
phi_shift = 10
u = 1e-5

X = 128
dx = 1
T = 50
dt = 5e-3
n_batches = 100
flat = False


label = 'phi_t_{}_l=2'.format(phi_t)

start_time = time.time()
solver = StoEvolutionPS(epsilon, a, k, u, phi_t, phi_shift)
solver.initialise(X, dx, T, dt, n_batches, radius=18, skew=5, flat=flat)
# solver.save_params(label)
solver.print_params()
solver.evolve(verbose=False)
# solver.save_phi(label)
end_time = time.time()
print('The simulation took: {}'.format(end_time - start_time))
