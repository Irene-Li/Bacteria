import time
import numpy as np
from ActiveModelAB import *
from StoEvolution2D import *


# Model parameters
epsilon = 0.2
phi_t = -0.4
phi_shift = 10
# delta = 10
u = 0

# run parameters and init
X = 128
n_batches = 100
flat = False 
radius = 35
T = 1e3

# things that don't change
a = 0.25
k = 1
dx = 1
dt = 1e-2
lbda = 1
zeta = 4
label = 'test'

solver = ActiveModelAB(epsilon, a, k, u, phi_t, phi_shift, lbda, zeta)
# solver = StoEvolution2D(epsilon, a, k, u, phi_t, phi_shift)


#solver.calculate_u(delta)
solver.initialise(X, dx, T, dt, n_batches, initial_value=phi_t, flat=flat, 
					radius=radius)
solver.save_params(label)
solver.print_params()

start_time = time.time()
solver.evolve(verbose=True, cython=False, fd=True)
end_time = time.time()

print(end_time-start_time)

solver.save_phi(label)
# solver.load(label)
solver.make_movie(label)
