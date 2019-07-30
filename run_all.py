import numpy as np
from FdEvolution import *
from multiprocessing import Pool

# parameters of the differential equation
a = 1e-1
k = 1
# simulation parameters
X = 400
dx = 0.1
dt = 1e-3
n_batches = 100

phi_shift = 100
phi_target = 0
phi_init = 0

alpha_tilde = a*(1 - 3*phi_target**2)
random = True


def run(delta):
	u_tilde = alpha_tilde**2/(4*k*(1+delta))
	u = u_tilde/(phi_target+phi_shift)
	T = 200/(u_tilde*delta)
	label = 'X_{}_delta_{}_amp_eq'.format(X, delta)
	print(label, u)
	solver = FdEvolution(a, k, u, phi_target, phi_shift)
	solver.initialise(X, dx, T, dt, n_batches, phi_init, flat=random)
	solver.evolve(verbose=True)
	solver.save(label)
	# solver.plot_steady_state(label)

deltas = [1e-1, 9e-2, 8e-2]
with Pool(len(deltas)) as p:
    print(p.map(run, deltas))
