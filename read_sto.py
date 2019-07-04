import numpy as np
from StoEvolutionPS import *


u = 5e-5
for phi_t in [-0.6]:
# for u in [5e-5, 6e-5, 7e-5, 8e-5]:
	label = 'phi_t_{}_u_{}_flat_2'.format(phi_t, u)
	solver = StoEvolutionPS()
	solver.load(label)
	solver.print_params()
	# solver.make_movie(label)
	solver.plot_slice(label, n=-1)
	# solver.make_bd_movie(label)
	# solver.plot_slices(label)
