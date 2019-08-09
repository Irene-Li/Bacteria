import time
import numpy as np
from DetEvolution2D import *
from StoEvolution2D import *

# Stochastic
sto = True

# Model parameters
delta = 0.1
epsilon = 0.001
phi_t = -0.25
phi_shift = 10

# run parameters and init
X = 128
n_batches = 100
flat = True
T = 1e4

# things that don't change
a = 0.2
k = 1
dx = 1
dt = 5e-3
alpha_tilde = a*(1 - 3*phi_t**2)
u_tilde = alpha_tilde**2/(4*k*(1+delta))
u = u_tilde/(phi_t+phi_shift)

label = 'sto_phi_t_{}_delta_{}'.format(phi_t, delta)
print(label, u)

if sto:
    solver = StoEvolution2D(epsilon, a, k, u, phi_t, phi_shift)
else:
    solver = DetEvolution2D(0, a, k, u, phi_t, phi_shift)

solver.initialise(X, dx, T, dt, n_batches, initial_value=phi_t, flat=flat)
solver.save_params(label)
solver.print_params()
solver.evolve(verbose=True)
solver.save_phi(label)
