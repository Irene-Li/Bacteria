import numpy as np
from StoEvolutionPS import *
from multiprocessing import Pool

# parameters of the differential equation
a = 0.1
k = 1
epsilon=0.01

# simulation parameters
X = 64
dx = 1
dt = 5e-3
n_batches = 100


phi_shift = 1.05
phi_target = -0.35
phi_init = phi_target
flat = False

def run(u):
	T = 0.1/u
	label = 'phi_t_{}_u_{}'.format(phi_target, u)
	print(label)
	solver = StoEvolutionPS(epsilon, a, k, u, phi_target, phi_shift)
	solver.initialise(X, dx, T, dt, n_batches, radius=10, skew=0, flat=flat)
	solver.evolve(verbose=True)
	solver.save(label)

u = [1e-5]
with Pool(len(u)) as p:
    print(p.map(run, u))
