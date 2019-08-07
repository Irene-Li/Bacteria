import time
import numpy as np
from TimeEvolution import *
from StoEvolution2D import *

twod = True

phi_t = 0
delta = 0.1
label = 'det_phi_t_{}_delta_{}'.format(phi_t, delta)

if twod:
    solver = StoEvolution2D()
else:
    solver = TimeEvolution()
solver.load(label)
solver.print_params()

if twod:
    # solver.make_move(label)
    solver.plot_slice(label, -1)
else:
    solver.plot_evolution(label, t_size=100, x_size=100)
