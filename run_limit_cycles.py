import numpy as np
from FdEvolution import *
from multiprocessing import Pool

# parameters of the differential equation
a = 1e-1
k = 1
# simulation parameters
X = 80
dx = 0.1
dt = 1e-3
n_batches = 1000

phi_shift = 1.05
phi_target = -0.35
phi_init = phi_target
flat = False

# def run(u):
# 	T = 10/u
# 	label = 'phi_t_{}_u_{}'.format(phi_target, u)
# 	print(label)
# 	solver = FdEvolution(a, k, u, phi_target, phi_shift)
# 	solver.initialise(X, dx, T, dt, n_batches, phi_init, flat=flat)
# 	solver.evolve(verbose=False)
# 	solver.save(label)
#
# u = [5e-6]
# with Pool(len(u)) as p:
#     print(p.map(run, u))

u = 0
n_batches = 100

def run_slow_mfold(phi_init):
    T = 1e6
    label = 'slow_mfold/phi_t_{}_init_{}'.format(phi_target, phi_init)
    print(label)
    solver = FdEvolution(a, k, u, phi_target, phi_shift)
    solver.initialise(X, dx, T, dt, n_batches, phi_init, flat=flat)
    solver.evolve(verbose=True)
    solver.save(label)

phi_init = [-0.705, -0.712]
with Pool(len(phi_init)) as p:
    print(p.map(run_slow_mfold, phi_init))
