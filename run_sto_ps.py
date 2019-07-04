import time
import numpy as np
from StoEvolutionPS import *

epsilon = 0.1
a = 0.2
k = 1
phi_t = -0.65
phi_shift = 10
u = 3e-5

X = 128
dx = 1
T = 2e4
dt = 5e-3
n_batches = 100
flat = False


label = 'phi_t_{}_u_{}'.format(phi_t, u)

start_time = time.time()
solver = StoEvolutionPS(epsilon, a, k, u, phi_t, phi_shift)
solver.initialise(X, dx, T, dt, n_batches, radius=17, skew=4, flat=flat)
solver.save_params(label)
solver.print_params()
solver.evolve(verbose=True, cython=True)
solver.save_phi(label)
end_time = time.time()
print('The simulation took: {}'.format(end_time - start_time))

# solver.make_movie(label)
