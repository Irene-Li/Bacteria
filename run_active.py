import time
import numpy as np
from ActiveModelAB import *
from StoEvolution2D import *


# Model parameters
epsilon = 0.05
phi_t = 0
phi_shift = 10
# delta = 10
u = 5e-5

# run parameters and init
X = 128
n_batches = 100
flat = True
T = 1e3

# things that don't change
a = 0.1
k = 1
dx = 1
dt = 0.001
lbda = 0
zeta = 0
label = 'test'

solver = ActiveModelAB(epsilon, a, k, u, phi_t, phi_shift, lbda, zeta)
# solver = StoEvolution2D(epsilon, a, k, u, phi_t, phi_shift)


# solver.calculate_u(delta)
solver.initialise(X, dx, T, dt, n_batches, initial_value=phi_t, flat=flat, 
					radius=10)
solver.save_params(label)
solver.print_params()
solver.evolve(verbose=True, cython=False)
solver.save_phi(label)
solver.make_movie(label)
