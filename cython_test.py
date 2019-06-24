from pseudospectral import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as am
import time
from scipy.integrate import ode
from scipy.fftpack import fft2, ifft2, fftfreq

epsilon = 0.1
a = 0.2
k = 1
u = 1e-5
M1 = 1
phi_t = -0.8
phi_shift = 10

X = 128
dx = 1
T = 50
dt = 5e-3
n_batches = 10
initial_value = -0.8
flat = False

x = np.arange(X)
y = np.arange(X)
x, y = np.meshgrid(x, y)
midpoint = int(X/2)
bubble_size = 25
l = np.sqrt(k/a)
init = fft2(- np.tanh((np.sqrt(1.2*(x-midpoint)**2+0.7*(y-midpoint)**2)-bubble_size)/l))

nitr = int(T/dt)
batch_size = int(nitr/n_batches)

start_time = time.time()
phi = evolve(init, a, k, u, phi_shift, phi_t, epsilon, dt, nitr, batch_size, X)
end_time = time.time()
print('The simulation took: {}'.format(end_time - start_time))

# fig = plt.figure()
# low, high = -1.2, 1.2
# ims = []
# im = plt.imshow(phi[0], vmin=low, vmax=high, animated=True)
# plt.colorbar(im)
# for i in range(n_batches):
# 	xy = phi[i]
# 	im = plt.imshow(xy, vmin=low, vmax=high, animated=True)
# 	ims.append([im])
# ani = am.ArtistAnimation(fig, ims, interval=100, blit=True,
# 								repeat_delay=1000)
# mywriter = am.FFMpegWriter()
# ani.save("movie.mp4", writer=mywriter)
