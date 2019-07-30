import time
import numpy as np
from FdEvolution import *

X = 300
deltas = np.array([1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2])
amp = []
n = []
for delta in deltas:
    label = 'X_{}_delta_{}_amp_eq'.format(X, delta)
    solver = FdEvolution()
    solver.load(label)
    solver.rescale_to_standard()
    # solver.plot_steady_state(label)
    amp.append(np.max(solver.phi[-2]))
    n.append(solver.count())

plt.plot(np.log(deltas), np.log(amp), 'x')
plt.plot(np.log(deltas), np.log(n), 'o')
plt.show()
