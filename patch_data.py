import time
import numpy as np
from PDE import *

# the labels of the runs that we want to patch together 
labels = ['target_54_2', 'target_54_3']

# the label we are going to use on the patch 
new_label = 'target_54_patch'

# read data from the runs 
tor = 1e-3

solver = TimeEvolution()
solver.load(labels[0])
phi = solver.phi
T = solver.T 
step_size = solver.step_size
n_batches = solver.n_batches

for label in labels[1:]:
	solver.load(label)
	assert np.abs(step_size - solver.step_size) < tor 
	phi = np.append(phi, solver.phi, axis=0)
	T += solver.T 
	n_batches += solver.n_batches

# abuse the class to plot what we want 
solver.phi = phi 
solver.T = T 
solver.n_batches = n_batches

t_size = 100
n_samples = 3
solver.plot_evolution(t_size, new_label)
solver.plot_average(new_label)
solver.plot_phi_bar_dot(new_label)
solver.plot_samples(new_label, n=n_samples)
solver.plot_phase_space(new_label)
print(solver.size)