import time
import numpy as np
from TimeEvolution import *
from StoEvolution2D import *

twod = False

phi_t = -0.25
delta = 0.1
u = 2e-4
label = 'u_{}_large'.format(u)

if twod:
    solver = StoEvolution2D()
else:
    solver = TimeEvolution()
solver.load(label)
# solver.print_params()

if twod:
    # solver.make_movie(label)
    # solver.plot_slice(label, -1)
    plt.plot(solver.phi[-1, 10])
    plt.show()
else:
    solver.phi = solver.phi[250:]
    solver.plot_evolution(label, t_size=250, x_size=300)
