import time
import numpy as np
from StoEvolution1D import *
from multiprocessing import Pool

# parameters of the differential equation
a = 0.1
k = 1
epsilon = 0.01
phi_shift = 10
phi_target = 0

# simulation parameters
X = 400
dx = 1
dt = 5e-3
n_batches = 100
flat = False
initial_value = phi_target
ps = True



def run(u):
    T = 2/u
    label = 'X_{}_u_{}_tanh'.format(X, u)
    print(label)
    start_time = time.time()
    solver = StoEvolution1D(epsilon, a, k, u, phi_target, phi_shift)
    solver.initialise(X, dx, T, dt, n_batches, initial_value=initial_value, flat=flat)
    solver.print_params()
    solver.evolve(verbose=True, ps=ps)
    solver.save(label)
    end_time = time.time()

us = [5e-6]
with Pool(len(us)) as p:
        print(p.map(run, us))
