import numpy as np
from StoEvolutionPS import *



for phi_t in [-0.6]:
# for u in [5e-5, 6e-5, 7e-5, 8e-5]:
	label = 'phi_t_{}_u_8e-05_l=2_3'.format(phi_t)
	solver = StoEvolutionPS()
	solver.load(label)
	solver.print_params()
	solver.make_movie(label)
	# solver.make_bd_movie(label)
	# solver.plot_slices(label)
