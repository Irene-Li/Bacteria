import time
import numpy as np
from FdEvolution import *

rates = [3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]

for u in rates:
	for init in ['flat', 'sin']:
		for length in [100, 200, 300]:
			label = '{}_X_{}_u_{}'.format(init, length, u)
			print(label)
			solver = FdEvolution()
			solver.load(label)
			solver.rescale_to_standard()
			solver.plot_steady_state(label, kink=0)
			solver.plot_evolution(100, 100, label)
			print(solver.count())
