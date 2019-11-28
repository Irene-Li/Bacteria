import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from StoEvolution2D import *

# plot amp eq
# slices = [10, 40, -1]
# phi_t = 0
# delta = 0.1
# epsilon = 1e-4
# # add_ons = ['', '', '_2']


# plot droplet splitting patterns
labels = ['StoPs2d/droplet_splitting/phi_t_-0.6_u_4e-05_X=256_12',
'StoPs2d/droplet_splitting/phi_t_-0.6_u_4e-05_X=256_14',
'StoPs2d/droplet_splitting/phi_t_-0.6_u_4e-05_X=256_17',
'2DLimitCycles/phi_t_-0.3_phi_s_1.5_u_8e-05_4']
slices = [10, -1, 10, -1]

ncols = len(slices)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)
fig = plt.figure(figsize=(ncols*4+0.3, 3.5))
grid = AxesGrid(fig, 111, nrows_ncols=(1, ncols),
                axes_pad=0.2,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.3)

for i in range(ncols):
    s = slices[i]
    ax = grid[i]
    # label = 'phi_t_{}_delta_{}_epsilon_{}{}'.format(phi_t, delta, epsilon, add_ons[i])
    label = labels[i]
    solver = StoEvolution2D()
    solver.load(label)
    phi = solver.phi[s]
    if i < 3:
        phi = np.roll(phi, 100, axis=-1)
        phi = (phi[::2, ::2] + phi[1::2, 0::2] + phi[0::2, 1::2] + phi[1::2, 1::2])/4
        print(phi.shape)
    ax.set_axis_off()
    im = ax.imshow(phi, vmin=-1, vmax=1, cmap='seismic', interpolation='none')

cbar = grid.cbar_axes[0].colorbar(im)

cbar.ax.set_yticks([-1, 0, 1])
cbar.ax.set_yticklabels([r'-$\phi_\mathrm{B}$', r'0', r'$\phi_\mathrm{B}$'])

plt.savefig('dumbbell.pdf')
plt.close()
