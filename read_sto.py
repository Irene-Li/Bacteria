import numpy as np
from StoEvolutionPS import *


for u in [1e-4, 5e-5]:
	label = 'u_{}_flat'.format(u)
	solver = StoEvolutionPS()
	solver.load(label)
	# solver.print_params()
	solver.make_movie(label)
