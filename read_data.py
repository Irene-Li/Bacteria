import time
import numpy as np
from TimeEvolution import *
from StoEvolution2D import *

twod = False

phi_t = 0
phi_s = 1.4
u = 5e-5
epsilon = 0.07
label = 'u_1e-07_epsilon_0.02_19'

if twod:
    solver = StoEvolution2D()
else:
    solver = StoEvolution1D()
solver.load(label)
# solver.print_params()

if twod:
    # solver.make_movie(label)
    solver.measure_domain(label)

else:
    solver.plot_steady_state(label)
    # solver.plot_evolution(label, t_size=100, x_size=512)
    # solver.plot_wavelength_evol(label)
