import numpy as np
from DetEvolution1D import *
import time
from multiprocessing import Pool

# parameters of the differential equation
a = 0.1
k = 1
phi_shift = 1.05
phi_target = -0.45

# simulation parameters
X = 600
dx = 1
dt = 1e-3
n_batches = 500
flat = False
initial_value = phi_target
ps = True

def run(u):
	T = 100/u
	label = 'u_{}_tanh'.format(u)
	print(label)
	start_time = time.time()
	solver = DetEvolution1D(a, k, u, phi_target, phi_shift)
	solver.initialise(X, dx, T, dt, n_batches, initial_value=initial_value, flat=flat)
	solver.evolve(verbose=True, ps=ps)
	solver.save(label)
	solver.plot_evolution(label, x_size=250, t_size=250)
	end_time = time.time()
	print(end_time - start_time)

run(1e-4)

# us = [1e-5]
# with Pool(2) as p:
#     print(p.map(run, us))
