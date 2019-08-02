import time
import numpy as np
from FdEvolution import *

phi_t = -0.35
rates = [5e-6]
for u in rates:
    label = 'phi_t_{}_u_{}'.format(phi_t, u)
    solver = FdEvolution()
    solver.load(label)
    # solver.rescale_to_standard()
    solver.plot_evolution(label, t_size=200, x_size=200)
