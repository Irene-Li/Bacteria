import time
import numpy as np
from StoEvolution import *
from multiprocessing import Pool

# parameters of the differential equation
a = 0.1
k = 1
epsilon = 0.05
phi_shift = 10
phi_target = 0
alpha_tilde = a*(1 - 3*phi_target**2)

# simulation parameters
X = 400
dx = 1
dt = 1e-3
n_batches = 500
flat = False
initial_value = phi_target



def run(delta):
    u_tilde = alpha_tilde**2/(4*k*(1+delta))
    u = u_tilde/(phi_target+phi_shift)
    T = 50/(u_tilde*delta)
    label = 'X_{}_delta_{}_sin'.format(X, delta)
    start_time = time.time()
    solver = StoEvolution(epsilon, a, k, u, phi_target, phi_shift)
    solver.initialise(X, dx, T, dt, n_batches, initial_value=initial_value, flat=flat)
    solver.print_params()
    solver.evolve(verbose=True)
    solver.save(label)
    end_time = time.time()

deltas = [1000, 3000]
with Pool(len(deltas)) as p:
        print(p.map(run, deltas))
