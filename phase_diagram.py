import numpy as np
from TimeEvolution import *
import matplotlib.pyplot as plt
import os

label = 'slow_mfold/phi_t_-0.35_init_-0.712'
solver = TimeEvolution()
solver.load(label)

# Take the last time slice
phi = solver.phi[-2]

# set the range of phi_shift and phi_target
phi_shift = np.arange(1.0, 1.8, 0.001)
phi_target = np.arange(-0.58, 0, 0.001)
x, y = np.meshgrid(phi_shift, phi_target)

# calculate the average rate of change
phi = phi[np.newaxis, np.newaxis, :]
x = x[:, :, np.newaxis]
y = y[:, :, np.newaxis]
rate = np.mean(- (phi + x) * (phi - y), axis=-1)
levels = [rate[0,0], 0, rate[-1,-1]]

# plot the phase diagram
x = np.squeeze(x)
y = np.squeeze(y)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)
plt.contourf(x, y, rate, levels, cmap=plt.cm.Blues)
plt.xlabel(r'$- \phi_\mathrm{a}$')
plt.ylabel(r'$\phi_\mathrm{t}$')
plt.text(1.13, -0.49, r'Cycles',  {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
plt.text(1.5, -0.25, r'Steady state',  {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
plt.tight_layout()
plt.savefig('phase_diagram.pdf')
plt.close()
