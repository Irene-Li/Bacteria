import time
import numpy as np
from DetEvolution2D import *
from StoEvolution2D import *

# Stochastic
sto = False

# Model parameters
epsilon = 0
phi_t = 0
phi_shift = 10
delta = 0.1
u = 0

# run parameters and init
X = 128
n_batches = 100
flat = False
T = 1e4

# things that don't change
a = 0.1
k = 1
dx = 1
dt = 1e-3

label = 'phi_t_{}_delta_{}_dt_{}'.format(phi_t, delta, dt)

if sto:
    solver = StoEvolution2D(epsilon, a, k, u, phi_t, phi_shift)
else:
    solver = DetEvolution2D(0, a, k, u, phi_t, phi_shift)

solver.calculate_u(delta)
solver.initialise(X, dx, T, dt, n_batches, initial_value=phi_t, flat=flat, radius=10)
solver.save_params(label)
solver.print_params()
solver.evolve(verbose=True)
solver.save_phi(label)
solver.make_movie(label)
