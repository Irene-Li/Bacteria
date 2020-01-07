import time
import numpy as np
from StoEvolution1D import *
from mkl_fft import fft

# us = [3.5e-6, 4e-6, 4.5e-6, 5e-6, 5.5e-6, 6e-6, 6.5e-6, 7e-6, 7.5e-6, 8e-6, 8.5e-6, 9e-6, 9.5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 3.5e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4]
us = [2e-5, 1e-5, 8e-6, 5e-6, 2e-6, 1e-6, 5e-7, 1e-7]
epsilons = [0.02, 0.01]
# epsilons = [0.05]
indices = ['_8', '_8', '_8', '_8', '_8', '_17', '_19', '_19']
# indices = ['_6', '_6', '_6', '_6', '_6', '_6', '_6']
length = 2048
average_over = 100
shape = (len(epsilons), len(us))
means = np.empty(shape, dtype='float64')
error_bars = np.zeros(shape, dtype='float64')

for (i, epsilon) in enumerate(epsilons):
	for (k, u) in enumerate(us):
		label = 'u_{}_epsilon_{}{}'.format(u, epsilon, indices[k])
		print(label)
		solver = StoEvolution1D()
		solver.load(label)
		# solver.plot_evolution(label, t_size=200, x_size=400)
		# solver.plot_wavenum_evol(label)
		phi = solver.phi[-average_over:]
		# from skimage.feature import canny
		# edges = canny(phi, sigma=10)
		# wavenumbers = np.sum(edges, axis=-1)[1:-1]/(2*length)
		cutoff = int(length/2)
		amplitudes = np.absolute(fft(phi)[0:cutoff])
		plt.plot(amplitudes[-1])
		plt.show()
		wavenumbers = np.argmax(amplitudes, axis=-1)
		means[i, k] = np.mean(wavenumbers)
		error_bars[i, k] = np.std(wavenumbers)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
qc = np.sqrt(solver.a/(2*solver.k))
x = np.log(us)
for (i, epsilon) in enumerate(epsilons):
	y = np.log(length) - np.log(means[i])
	poly = np.poly1d(np.polyfit(x, y, 1))
	plt.errorbar(x, y, yerr=error_bars[i]/means[i], fmt='x', label=r'$\epsilon={{{}}}$'.format(epsilon))
	plt.plot(x, poly(x), '--', label=r'gradient={:.4f}'.format(poly.c[0]))

plt.xlabel(r'$\log(M_\mathrm{A} u )$')
plt.ylabel(r'$\log(L)$')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('pattern_length.pdf')
plt.close()
#
# print(2*np.pi/qc)
