import numpy as np
from StoEvolutionPS import *


u = 5e-5
for phi_t in [-0.65]:
# for u in [5e-5, 6e-5, 7e-5, 8e-5]:
	label = 'phi_t_{}_u_{}_flat_4'.format(phi_t, u)
	solver = StoEvolutionPS()
	solver.load(label)
	solver.print_params()
	solver.make_movie(label)
	# solver.make_bd_movie(label)
	# solver.plot_slices(label)
