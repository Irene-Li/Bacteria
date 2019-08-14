import time
import numpy as np
from TimeEvolution import *
from StoEvolution2D import *

twod = True

phi_t = -0.5
delta = 0.1
u = 5e-5
label = 'phi_t_{}_u_{}_5'.format(phi_t, u)

if twod:
    solver = StoEvolution2D()
else:
    solver = TimeEvolution()
solver.load(label)
# solver.print_params()

if twod:
    # solver.make_movie(label)
    solver.plot_phase_space(label)
    solver.plot_average(label)

else:
    solver.phi = solver.phi[250:]
    solver.plot_evolution(label, t_size=250, x_size=300)
