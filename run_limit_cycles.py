import numpy as np
from StoEvolution2D import *
from multiprocessing import Pool

# parameters of the differential equation
a = 0.2
k = 1
epsilon = 0.01

# simulation parameters
X = 128
dx = 1
dt = 5e-3
T = 2e4
n_batches = 100


phi_shift = 1.1
phi_target = -0.5
phi_init = phi_target
flat = False

def run(u):
	label = 'phi_t_{}_u_{}'.format(phi_target, u)
	print(label)
	solver = StoEvolution2D(epsilon, a, k, u, phi_target, phi_shift)
	solver.initialise(X, dx, T, dt, n_batches, radius=15, skew=4, flat=flat)
	solver.evolve(verbose=True)
	solver.save(label)
	solver.make_movie(label)

run(5e-5)
