import time
import numpy as np
from DetEvolution1D import *

us = [1e-5, 8e-6, 9e-6, 7e-6, 6e-6, 3e-6, 5e-6, 2e-6]
Ls = [1000, 2000]
us = np.sort(us)
inits = ['tanh']
n = np.zeros((len(Ls), len(inits), len(us)))
for (i, length) in enumerate(Ls):
	for (j, init) in enumerate(inits):
		for (k, u) in enumerate(us):
			label = 'X={}/u_{}_{}'.format(length, u, init)
			print(label)
			solver = DetEvolution1D()
			solver.load(label)
			# solver.plot_steady_state(label)
			n[i, j, k] = solver.count()

qc = np.sqrt(solver.a/(2*solver.k))
x = np.log(us)
for (i, length) in enumerate(Ls):
	for (j, init) in enumerate(inits):
		y = np.log(length/n[i,j])
		poly = np.poly1d(np.polyfit(x, y, 1))
		plt.plot(x, y, 'x', label='X={}, {}'.format(length, init))
		plt.plot(x, poly(x), '--')
		print(poly.c[0], length, init)
plt.axhline(y=np.log(2*np.pi/qc), color='k', label=r'$2\pi/q_\mathrm{c}$')
plt.legend()
plt.savefig('pattern.pdf')
plt.close()
