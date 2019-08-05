import time
import numpy as np
from DetEvolution1D import *

us = [1e-5, 8e-6, 9e-6, 7e-6, 6e-6, 3e-6, 5e-6, 2e-6]
Ls = [600, 800]
us = np.sort(us)
inits = ['flat', 'tanh']
n = np.zeros((len(Ls), len(inits), len(us)))
for (i, length) in enumerate(Ls):
	for (j, init) in enumerate(inits):
		for (k, u) in enumerate(us):
			label = 'X={}/u_{}_{}'.format(length, u, init)
			print(label)
			solver = DetEvolution1D()
			solver.load(label)
			# solver.rescale_to_standard()
			# solver.plot_steady_state(label)
			# solver.plot_evolution(label, 200, 200)
			n[i, j, k] = solver.count()

x = np.log(us)
for (i, length) in enumerate(Ls):
	for (j, init) in enumerate(inits):
		y = np.log(length/n[i,j])
		poly = np.poly1d(np.polyfit(x, y, 1))
		plt.plot(x, y, 'x', label='X={}, {}'.format(length, init))
		plt.plot(x, poly(x), '--')
		print(poly.c[0], length, init)
plt.legend()
plt.savefig('pattern.pdf')
plt.close()
