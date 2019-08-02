import time
import numpy as np
from FdEvolution import *

X = 300
deltas = [0.04, 0.1]


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
for delta in deltas:
    label = 'X_{}_delta_{}_amp_eq'.format(X, delta)
    solver = FdEvolution()
    solver.load(label)
    # solver.rescale_to_standard()
    phi = solver.phi[-2]
    plt.plot(phi, label='$\Delta={}$'.format(delta))


plt.xticks([])
plt.yticks([-1, 0, 1],
            [r'$\phi_\mathrm{B}$', r'$\phi_\mathrm{t}$', r'$-\phi_\mathrm{B}$'])
plt.ylim([-1.1, 1.1])
plt.xlabel(r'$x$')
plt.legend()
plt.savefig('amp_eq.pdf')
plt.close()
