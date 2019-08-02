import time
import numpy as np
from StoEvolutionPS import *
from multiprocessing import Pool

epsilon = 0.1
a = 0.2
k = 1
phi_t = 0
phi_shift = 10

X = 64
dx = 1
dt = 5e-3
n_batches = 100
flat = True

alpha_tilde = a*(1 - 3*phi_t**2)

def run(delta):
    u_tilde = alpha_tilde**2/(4*k*(1+delta))
    u = u_tilde/(phi_t+phi_shift)
    T = 1/(u_tilde*delta)
    label = 'X_{}_delta_{}'.format(X, delta)
    print(label, u)
    solver = StoEvolutionPS(epsilon, a, k, u, phi_t, phi_shift)
    solver.initialise(X, dx, T, dt, n_batches, initial_value=phi_t, flat=flat)
    solver.save_params(label)
    solver.print_params()
    solver.evolve(verbose=True, cython=True)
    solver.save_phi(label)

deltas = [0.1, 0.01]
with Pool(len(deltas)) as p:
    print(p.map(run, deltas))
