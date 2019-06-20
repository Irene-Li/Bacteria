import time
import numpy as np
from StoEvolution import *
from multiprocessing import Pool

# parameters of the differential equation
a = 0.2
k = 1
epsilon = 0.01
u = 1e-5
phi_shift = 10
phi_target = 0

# simulation parameters
X = 100
dx = 1
dt = 1e-3
T = 1e5
n_batches = 100
flat = True
initial_value = phi_target

def run(u):
    label = 'u_{}_ep_{}'.format(u, epsilon)
    start_time = time.time()
    solver = StoEvolution(epsilon, a, k, u, phi_target, phi_shift)
    solver.initialise(X, dx, T, dt, n_batches, initial_value=initial_value, flat=flat)
    solver.print_params()
    solver.evolve(verbose=False)
    solver.save(label)
    end_time = time.time()

rates = [1e-5, 5e-5, 1e-4]
with Pool(len(rates)) as p:
        print(p.map(run, rates))
