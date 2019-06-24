import numpy as np
from StoEvolution import *

epsilon = 0.01
rates = [5e-5, 1e-4]
for u in rates:
    label = 'u_{}_ep_{}'.format(u, epsilon)
    solver = StoEvolution()
    solver.load(label)
    solver.plot_evolution(label, t_size=100, x_size=200)
