import time
import numpy as np
from StoEvolution import *

for dt in [1e-3, 1e-4]:
	label = 'sto_dt_{}_long'.format(dt)
	solver = StoEvolution()
	solver.load(label)
	solver.rescale_to_standard()
	solver.plot_steady_state(label, kink=0)
	solver.plot_evolution(100, 100, label)
