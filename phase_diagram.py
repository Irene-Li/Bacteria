import numpy as np
from FdEvolution import *
import matplotlib.pyplot as plt
import os

label = 'init_-0.802'
solver = FdEvolution()
solver.load(label)
solver.rescale_to_standard()

# Take the last time slice 
phi = solver.phi[-1] 

# set the range of phi_shift and phi_target
phi_shift = np.arange(1.0, 1.8, 0.01)
phi_target = np.arange(-0.58, 0, 0.01)
x, y = np.meshgrid(phi_shift, phi_target)

# calculate the average rate of change 
phi = phi[np.newaxis, np.newaxis, :]
x = x[:, :, np.newaxis]
y = y[:, :, np.newaxis]
rate = np.mean(- (phi + x) * (phi - y), axis=-1)

# plot the phase diagram 
x = np.squeeze(x)
y = np.squeeze(y)
levels = [rate[0, 0], -0.01, rate[-1, -1]]

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
plt.contourf(x, y, rate, levels, cmap=plt.cm.Blues)
plt.xlabel(r'$\phi_\mathrm{ref}$')
plt.ylabel(r'$\phi_\mathrm{t}$')
plt.text(1.13, -0.49, r'Cycles',  {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
plt.text(1.5, -0.25, r'Steady state',  {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
plt.title(r'phase diagram')
plt.tight_layout()
plt.savefig('phase_diagram.pdf')


