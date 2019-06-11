import numpy as np
from StoEvolutionPS import *



for phi_t in [-0.6, -0.7, -0.8]:
# for u in [5e-6, 1e-5, 5e-5, 1e-4]:
	label = 'phi_t_{}_skewed_droplet'.format(phi_t)
	solver = StoEvolutionPS()
	solver.load(label)
	solver.print_params()
	# solver.make_movie(label)
	# solver.plot_slices(label)
