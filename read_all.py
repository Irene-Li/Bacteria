import time
import numpy as np
from FdEvolution import *

rates = [3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]

for u in rates:
	for init in ['flat', 'sin']:
		label = '{}_X_200_u_{}'.format(init, u)
		print(label)
		solver = FdEvolution()
		solver.load(label)
		solver.rescale_to_standard()
		solver.plot_steady_state(label, kink=0)
		solver.plot_evolution(100, 100, label)
		print(solver.count())
