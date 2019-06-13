import time
import numpy as np
from StoEvolutionPS import *

epsilon = 0.05
a = 0.2
k = 1
u = 3e-5
phi_t = -0.7
phi_shift = 10

X = 256
dx = 1
T = 1e4
dt = 5e-3
n_batches = 100
flat = False


label = 'phi_t_{}_l=2_X=256'.format(phi_t)

start_time = time.time()
solver = StoEvolutionPS(epsilon, a, k, u, phi_t, phi_shift)
solver.initialise(X, dx, T, dt, n_batches, radius=20, skew=5, flat=flat)
solver.save_params(label)
solver.print_params()
solver.evolve(verbose=True)
solver.save_phi(label)
end_time = time.time()
print('The simulation took: {}'.format(end_time - start_time))
