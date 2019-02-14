import time
import numpy as np
from PDE import * 

# the label for this run 
label = 'init_phi_s'

# parameters of the differential equation
a = 1e-1
b = 1e-1
k = 1

# spinodal, binodal and target densities
phi_b = np.sqrt(a/b)
phi_s = np.sqrt(a/(3 * b))

# simulation parameters
X = 60
dx = 0.01
dt = 5e-3
T = dt * 1e5
n_batches = 1e3

# Evolve
u = 0
phi_shift = phi_b
phi_target = 0 
phi_init = - phi_s  
b_gradient = 0 

start_time = time.time()
solver = PsEvolution(a, b, k, u, phi_target, phi_shift, b_gradient=b_gradient)
solver.initialise(X, dx, T, dt, n_batches, phi_init)
solver.make_shifted_interface(phi_init)
solver.evolve()
solver.save(label)
end_time = time.time()
print('The simulation took: ')
print(end_time - start_time)

t_size = 100
n_samples = 4
solver.plot_evolution(t_size, label)
solver.plot_samples(label, n=n_samples)
solver.plot_average(label)
