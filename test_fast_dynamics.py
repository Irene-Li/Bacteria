import numpy as np
from PDE import *
import matplotlib.pyplot as plt


label = 'a_3e-2_cont'
tol = 1e-3

# Extract the samples where phi_bar_dot is near zero 
solver = FdEvolution()
solver.load(label)
phi_bar_dot = np.mean(-(solver.phi[:, 2:-2] - solver.phi_target) * (solver.phi[:, 2:-2] + solver.phi_shift), axis=1)
phi = solver.phi[(phi_bar_dot < tol) & (phi_bar_dot > -tol)]

print('Number of samples within the tolerance:', phi.shape[0])

# solver.plot_phi(label, phi)

# Evolve the sample under the fast dynamics 
new_label = 'fast_dynamics'

solver.u = 0 
solver.T = solver.dt * 1e7
solver.n_batches = int(1e2)
solver.step_size = solver.T/solver.n_batches
solver.batch_size = int(solver.step_size/solver.dt)

phi = phi[int(phi.shape[0]/2 - 0.5)]
solver.phi_initial = phi 
solver.evolve()
solver.save(new_label)


