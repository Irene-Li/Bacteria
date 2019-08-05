import time
import numpy as np
from DetEvolution1D import *

us = [5e-5, 2e-5, 1e-5, 8e-6, 5e-6]
Ls = [600, 800]
inits = ['flat', 'tanh']
n = np.zeros((len(Ls), len(inits), len(us)))
for (i, length) in enumerate(Ls):
	for (j, init) in enumerate(inits):
		for (k, u) in enumerate(us):
			label = 'X={}/X_{}_u_{}_{}_ps'.format(length, length, u, init)
			print(label)
			solver = DetEvolution1D()
			solver.load(label)
			# solver.rescale_to_standard()
			# solver.plot_steady_state(label)
			# solver.plot_evolution(label, 200, 200)
			n[i, j, k] = solver.count()

for (i, length) in enumerate(Ls):
	for (j, init) in enumerate(inits):
		plt.plot(np.log(us), np.log(length/n[i,j]), 'x', label='X={}, {}'.format(length, init))
plt.legend()
plt.savefig('pattern.pdf')
plt.close()
