import time
import numpy as np
from StoEvolution import *
from multiprocessing import Pool

# parameters of the differential equation
a = 0.05
k = 1
epsilon = 0.01
phi_shift = 10
phi_target = 0

# simulation parameters
X = 200
dx = 1
dt = 1e-3
n_batches = 100
flat = False
initial_value = phi_target

def run(u):
    T = 5/(u*phi_shift)
    label = 'u_{}_ep_{}_sin'.format(u, epsilon)
    start_time = time.time()
    solver = StoEvolution(epsilon, a, k, u, phi_target, phi_shift)
    solver.initialise(X, dx, T, dt, n_batches, initial_value=initial_value, flat=flat)
    solver.print_params()
    solver.evolve(verbose=False)
    solver.save(label)
    end_time = time.time()

rates = [3e-4, 1e-4, 3e-5, 1e-5, 3e-6]
with Pool(len(rates)) as p:
        print(p.map(run, rates))
