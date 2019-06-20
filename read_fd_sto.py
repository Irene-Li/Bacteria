import numpy as np
from StoEvolution import *


label = 'u_1e-05_ep_0.01'
solver = StoEvolution()
solver.load(label)
solver.plot_evolution(100, 100, label)
# solver.make_movie(label)
# solver.make_bd_movie(label)
# solver.plot_slices(label)
