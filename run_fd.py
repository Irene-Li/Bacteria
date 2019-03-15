import time
import numpy as np
from FdEvolution import *

# the label for this run
label = 'X_50_u_1e-6'

# parameters of the differential equation
a = 0.5
k = 1

# simulation parameters
X = 50
dx = 0.1
dt = 1e-3
n_batches = 100

# Evolve
u = 1e-6
T = 1/u
phi_shift = 100
phi_target = 0
phi_init = 0
random = False

start_time = time.time()
solver = FdEvolution(a, k, u, phi_target, phi_shift)
solver.initialise(X, dx, T, dt, n_batches, phi_init, random=random)
solver.evolve()
solver.save(label)
end_time = time.time()
print('The simulation took: ')
print(end_time - start_time)
