import time
import numpy as np
from DetEvolution1D import *
from mkl_fft import fft

us = [3.5e-6, 4e-6, 4.5e-6, 5e-6, 5.5e-6, 6e-6, 6.5e-6, 7e-6, 7.5e-6, 8e-6, 8.5e-6, 9e-6, 9.5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 3.5e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4]
Ls = [800]
us = np.sort(us)
inits = ['flat']
average_over = 50
means = np.empty((len(Ls), len(inits), len(us)), dtype='float64')
error_bars = np.zeros((len(Ls), len(inits), len(us)), dtype='float64')
for (i, length) in enumerate(Ls):
	for (j, init) in enumerate(inits):
		for (k, u) in enumerate(us):
			label = 'X={}/u_{}_{}'.format(length, u, init)
			print(label)
			solver = DetEvolution1D()
			solver.load(label)
			# solver.plot_evolution(label, t_size=200, x_size=400)
			phi = solver.phi[-2-average_over:-2]
			amplitudes = np.absolute(fft(phi)[:, :int(length/2)])
			wavelengths = np.argmax(amplitudes, axis=-1)/length
			means[i,j,k] = np.mean(wavelengths)
			error_bars[i,j,k] = np.std(wavelengths)
			print(error_bars[i,j,k])

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
qc = np.sqrt(solver.a/(2*solver.k))
x = np.log(us)
for (i, length) in enumerate(Ls):
	for (j, init) in enumerate(inits):
		y = np.log(means[i,j])
		poly = np.poly1d(np.polyfit(x, y, 1))
		plt.errorbar(x, y, yerr=error_bars[i,j]/means[i,j], fmt='x', label=r'$X={}$, {}'.format(length, init))
		plt.plot(x, poly(x), '--', label=r'gradient={:.2f}'.format(poly.c[0]))
plt.legend()
plt.savefig('pattern_length.pdf')
plt.close()
#
# print(2*np.pi/qc)
