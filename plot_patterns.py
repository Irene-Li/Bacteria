import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from StoEvolutionPS import *

slices = [-1, -1, -1]
names = ['phi_t_-0.6_u_5e-05_flat_2', 'phi_t_0_u_5e-05_X=256_2',
            'phi_t_0.3_u_1e-05_2']

ncols = len(slices)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
fig = plt.figure(figsize=(8, 2.3))
grid = AxesGrid(fig, 111, nrows_ncols=(1, ncols),
                axes_pad=0.1,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1)

for i in range(ncols):
    s = slices[i]
    ax = grid[i]
    label = names[i]
    solver = StoEvolutionPS()
    solver.load(label)
    phi = solver.phi[s]
    ax.set_axis_off()
    im = ax.imshow(phi, vmin=-1, vmax=1, cmap='seismic')

cbar = grid.cbar_axes[0].colorbar(im)

cbar.ax.set_yticks([-1, 0, 1])
cbar.ax.set_yticklabels([r'-$\phi_\mathrm{B}$', r'0', r'$\phi_\mathrm{B}$'])

plt.savefig('patterns.pdf')
plt.close()
