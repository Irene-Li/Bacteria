import time
import numpy as np
from DetEvolution1D import *


for u in [8e-6]:
	label = 'X_400_u_{}_flat_ps'.format(u)
	solver = DetEvolution1D()
	solver.load(label)
	solver.plot_steady_state(label)
	solver.plot_evolution(label, x_size=200, t_size=100)
