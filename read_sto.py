import numpy as np
from StoEvolutionPS import *


label = 'u_2e-05_small_droplet_large_ep'
solver = StoEvolutionPS()
solver.load(label)
# solver.print_params()
solver.make_movie(label)
# solver.plot_slices(label)
