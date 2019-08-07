import time
import numpy as np
from StoEvolution1D import *
from multiprocessing import Pool

# parameters of the differential equation
a = 0.1
k = 1
epsilon = 0.02
phi_shift = 10
phi_target = 0

# simulation parameters
X = 400
dx = 1
dt = 5e-3
n_batches = 100
flat = True
initial_value = phi_target
ps = True



def run(u):
    T = 5/u
    label = 'u_{}_flat'.format(u)
    print(label)
    start_time = time.time()
    solver = StoEvolution1D(epsilon, a, k, u, phi_target, phi_shift)
    solver.initialise(X, dx, T, dt, n_batches, initial_value=initial_value, flat=flat)
    solver.print_params()
    solver.evolve(verbose=True, ps=ps)
    solver.save(label)
    solver.plot_evolution(label, t_size=200, x_size=200)
    end_time = time.time()
    print('time: {}'.format(end_time - start_time))

# us = [5e-6]
# with Pool(len(us)) as p:
#     print(p.map(run, us))

run(2e-5)
