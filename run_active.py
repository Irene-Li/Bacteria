import time
import numpy as np
from ActiveModelAB import *
from StoEvolution2D import *


# Model parameters
epsilon = 0.2
phi_t = -0.8
phi_shift = 10
# delta = 10
u = 1e-6

# run parameters and init
X = 128
n_batches = 50
flat = False 
radius = 60
T = 1e2
droplet = False 

# things that don't change
a = 0.25
k = 1
dx = 1
dt = 0.01
lbda = 0
zeta = 0
label = 'test'

solver = ActiveModelAB(epsilon, a, k, u, phi_t, phi_shift, lbda, zeta)
# solver = StoEvolution2D(epsilon, a, k, u, phi_t, phi_shift)


#solver.calculate_u(delta)
solver.initialise(X, dx, T, dt, n_batches, initial_value=phi_t, flat=flat, 
					radius=radius, skew=2, droplet=droplet)
solver.save_params(label)
solver.print_params()

start_time = time.time()
solver.evolve(verbose=True, cython=False, fd=True)
end_time = time.time()

print(end_time-start_time)

solver.save_phi(label)
# solver.load(label)
solver.make_movie(label)
