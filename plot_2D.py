from matplotlib import pyplot as plt
from matplotlib import animation as am
from scipy.io import savemat, loadmat
import numpy as np

Ng = 128
phi0 = 0.4
a = -0.25
label = 'N128__u00.00_a-0.2500'
M = loadmat(label + '_DA.mat')
evol = M['X']
evol = evol[::10]

fig = plt.figure()
ims = []
for i in range(evol.shape[0]):
    xy = evol[i].reshape((Ng, Ng))
    im = plt.imshow(xy, animated=True)
    ims.append([im])
ani = am.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=1000)
mywriter = am.FFMpegWriter()
ani.save(label+"movie.mp4", writer=mywriter)
