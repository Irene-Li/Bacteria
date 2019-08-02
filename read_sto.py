import numpy as np
from StoEvolutionPS import *

delta = 0.2
X = 128
phi_t = -0.2

label = 'X_{}_delta_{}'.format(X, delta)
solver = StoEvolutionPS()
solver.load(label)
solver.print_params()
solver.make_movie(label)
solver.plot_slice(label, n=-1)
# solver.make_bd_movie(label)
# solver.plot_slices(label)

phi_average = np.amax(solver.phi, axis=(1,2))
plt.plot(phi_average)
plt.show()
