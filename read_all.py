import time
import numpy as np
from StoEvolution import *

ep = 0.01
for u in [3e-4, 1e-4, 3e-5, 1e-5, 3e-6]:
	label = 'u_{}_ep_{}_sin'.format(u, ep)
	solver = StoEvolution()
	solver.load(label)
	solver.rescale_to_standard()
	solver.plot_steady_state(label, kink=0)
	solver.plot_evolution(label, x_size=200, t_size=100)
