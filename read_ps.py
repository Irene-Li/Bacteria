import time
import numpy as np
from PsEvolution import *

for phi_t in [-0.6]:
	label = 'phi_t_{}_l=2'.format(phi_t)
	solver = PsEvolution()
	solver.load(label)
	# solver.print_params()
	solver.make_movie(label)
