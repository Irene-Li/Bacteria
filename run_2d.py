import time
import numpy as np
from DetEvolution2D import *
from StoEvolution2D import *

# Stochastic
sto = False

# Model parameters
epsilon = 0.001
phi_t = -0.5
phi_shift = 1.05
u = 2e-4

# run parameters and init
X = 128
n_batches = 100
flat = False
T = 1e2

# things that don't change
a = 0.2
k = 1
dx = 1
dt = 5e-3

label = 'phi_t_{}_phi_s_{}'.format(phi_t, phi_shift)
print(label, u)

if sto:
    solver = StoEvolution2D(epsilon, a, k, u, phi_t, phi_shift)
else:
    solver = DetEvolution2D(0, a, k, u, phi_t, phi_shift)

solver.initialise(X, dx, T, dt, n_batches, radius=15, skew=2, flat=flat)
solver.save_params(label)
solver.print_params()
solver.evolve(verbose=True)
solver.save_phi(label)
solver.make_movie(label)
