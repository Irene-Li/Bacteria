import numpy as np
from StoEvolution2D import *

delta = 0.1
X = 128
phi_t = -0.35
u = 1e-5

label = 'phi_t_{}_u_{}'.format(phi_t, u)
solver = StoEvolution2D()
solver.load(label)
solver.print_params()
solver.make_movie(label)
# solver.plot_slice(label, n=-1)
# solver.make_bd_movie(label)
# solver.plot_slices(label)

# phi_average = np.amax(solver.phi, axis=(1,2))
# plt.plot(phi_average)
# plt.show()
