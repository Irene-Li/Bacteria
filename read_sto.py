import numpy as np
from StoEvolution import *

labels = ['sto_dt_1e-05']
t_size = 100
x_size = 200


for label in labels:
	solver = StoEvolution()
	solver.load(label)
	solver.rescale_to_standard()
	solver.load_phi(label)
	# solver.load_trajs(label)
	# solver.plot_fourier_components(label)
	solver.plot_evolution(t_size, x_size,label)

	solver.plot_average(label)
	# # solver.compute_mu(label, n=n_samples)
	# # solver.plot_phi_bar_dot(label)

	# solver.plot_free_energy(label)
	# solver.plot_phase_space(label)
	# solver.plot_samples(label, n=n_samples)
