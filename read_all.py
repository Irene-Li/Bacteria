import time
import numpy as np
from DetEvolution1D import *


for u in [5e-6, 2e-6]:
	label = 'u_{}_tanh_short'.format(u)
	solver = DetEvolution1D()
	solver.load(label)
	solver.plot_steady_state(label)
	solver.plot_evolution(label, x_size=400, t_size=400)
