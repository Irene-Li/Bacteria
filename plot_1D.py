import numpy as np
import matplotlib.pyplot as plt
from TimeEvolution import *

label = 'X_50_u_1e-4'

solver = TimeEvolution()
solver.load(label)
phi = solver.phi[-2]
x = np.arange(phi.size)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)
plt.fill_between(x, phi, y2=-1, alpha=0.2, edgecolor='MidnightBlue', facecolor='MidnightBlue')
plt.xlabel(r'$x$')
plt.xlim([0, phi.size])
plt.ylim([-1, 1])
plt.yticks([-1, 0, 1], [r'-$\phi_\mathrm{B}$', r'0', r'$\phi_\mathrm{B}$'])
plt.xticks([],[])
plt.savefig('ModelAB.pdf')
plt.close()
