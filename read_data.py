import time
import numpy as np
from FdEvolution import *

label = 'small_box_u_1e-6'

solver = FdEvolution()
solver.load(label)
solver.rescale_to_standard()

#solver.plot_average(label)
# solver.compute_mu(label)
# # solver.plot_phi_bar_dot(label)

# solver.plot_current(label)


# solver.plot_phase_space(label)
solver.plot_steady_state(label, kink=0)


# solver.comparison()
