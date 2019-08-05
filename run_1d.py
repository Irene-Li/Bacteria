import numpy as np
from DetEvolution1D import *
from multiprocessing import Pool

# parameters of the differential equation
a = 0.1
k = 1
phi_shift = 10
phi_target = 0

# simulation parameters
X = 800
dx = 1
dt = 1e-3
n_batches = 100
flat = False
initial_value = phi_target
ps = True


def run(u):
	T = 10/u
	label = 'u_{}_tanh'.format(u)
	print(label)
	start_time = time.time()
	solver = DetEvolution1D(a, k, u, phi_target, phi_shift)
	solver.initialise(X, dx, T, dt, n_batches, initial_value=initial_value, flat=flat)
	solver.evolve(verbose=False, ps=ps)
	solver.save(label)
	end_time = time.time()
	print(end_time - start_time)

us = [1e-6, 2e-6]
with Pool(len(us)) as p:
    print(p.map(run, us))
