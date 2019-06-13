import numpy as np
from StoEvolutionPS import *



for phi_t in [-0.7]:
# for u in [1e-5, 5e-5]:
	label = 'phi_t_{}_l=2_X=256'.format(phi_t)
	solver = StoEvolutionPS()
	solver.load(label)
	solver.print_params()
	solver.make_movie(label)
	# solver.plot_slices(label)
