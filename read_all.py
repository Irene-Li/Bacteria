import time
import numpy as np
from DetEvolution1D import *

X = 400
for u in [5e-5, 2e-5, 1e-5, 8e-6, 5e-6]:
	label = 'X_{}_u_{}_tanh_ps'.format(X, u)
	solver = DetEvolution1D()
	solver.load(label)
	# solver.rescale_to_standard()
	solver.plot_steady_state(label)
	solver.plot_evolution(label, x_size=200, t_size=100)
