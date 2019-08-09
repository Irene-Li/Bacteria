import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from StoEvolution2D import *

slices = [-1, -1]
names = ['det_phi_t_0_delta_0.1_4', 'sto_phi_t_0_delta_0.1_2']

ncols = len(slices)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
fig = plt.figure(figsize=(ncols*5+3, 5))
grid = AxesGrid(fig, 111, nrows_ncols=(1, ncols),
                axes_pad=0.5,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.3)

for i in range(ncols):
    s = slices[i]
    ax = grid[i]
    label = names[i]
    solver = StoEvolution2D()
    solver.load(label)
    phi = solver.phi[s]
    ax.set_axis_off()
    im = ax.imshow(phi, vmin=-1, vmax=1, cmap='seismic')

cbar = grid.cbar_axes[0].colorbar(im)

cbar.ax.set_yticks([-1, 0, 1])
cbar.ax.set_yticklabels([r'-$\phi_\mathrm{B}$', r'0', r'$\phi_\mathrm{B}$'])

plt.savefig('amp_eq_patterns.pdf')
plt.close()
