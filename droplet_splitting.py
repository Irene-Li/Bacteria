import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from StoEvolution2D import *

u = 4e-5
phi_t = -0.6

slices = [50, 50, 25, 0]
names = ['', '_2', '_3', '_4']
phi_collect = []

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
fig = plt.figure(figsize=(8, 2.3))
grid = AxesGrid(fig, 111, nrows_ncols=(1, 4),
                axes_pad=0.1,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1)

for i in range(4):
    s = slices[i]
    n = names[i]
    ax = grid[i]
    label = 'phi_t_{}_u_{}'.format(phi_t, u) + n
    solver = StoEvolution2D()
    solver.load(label)
    phi = solver.phi[s]
    phi_collect.append(solver.phi)
    ax.set_axis_off()
    im = ax.imshow(phi, vmin=-1, vmax=1, cmap='seismic')

solver.phi = np.concatenate(phi_collect, axis=0)
solver.phi = solver.phi[::4]
print(solver.phi.shape)
solver.make_movie('phi_t_{}_u_{}_all'.format(phi_t, u))

cbar = grid.cbar_axes[0].colorbar(im)

cbar.ax.set_yticks([-1, 0, 1])
cbar.ax.set_yticklabels([r'-$\phi_\mathrm{B}$', r'0', r'$\phi_\mathrm{B}$'])

plt.savefig('droplet_splitting.pdf')
plt.close()
