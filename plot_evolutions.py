import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from TimeEvolution import *

u = 2e-4
names = ['u_{}_large'.format(u), 'u_{}_tanh'.format(u)]

ncols = len(names)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
fig = plt.figure(figsize=(ncols*5+3, 5))
grid = AxesGrid(fig, 111, nrows_ncols=(1, ncols),
                axes_pad=0.5,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.3)

grid[0].set_ylabel(r'$t$')
for i in range(ncols):
    ax = grid[i]
    label = names[i]
    solver = TimeEvolution()
    solver.load(label)
    phi = np.flip(solver.phi[250:, ::2], axis=0)
    im = ax.imshow(phi, vmin=-1, vmax=1, cmap='seismic')
    ax.set_xlabel(r'$x$')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

cbar = grid.cbar_axes[0].colorbar(im)


cbar.ax.set_yticks([-1, 0, 1])
cbar.ax.set_yticklabels([r'-$\phi_\mathrm{B}$', r'0', r'$\phi_\mathrm{B}$'])

plt.savefig('evolutions.pdf')
plt.close()
