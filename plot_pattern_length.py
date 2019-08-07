import time
import numpy as np
from DetEvolution1D import *

us = [1e-5, 8e-6, 9e-6, 7e-6, 6e-6, 5e-6, 3e-6, 2e-6]
Ls = [1000, 2000]
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
			# solver.plot_steady_state(label)
			n[i, j, k] = solver.count()

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
qc = np.sqrt(solver.a/(2*solver.k))
x = np.log(us)
for (i, length) in enumerate(Ls):
	for (j, init) in enumerate(inits):
		y = (length/n[i,j])
		# poly = np.poly1d(np.polyfit(x, y, 1))
		plt.plot(x, y, 'x', label=r'$X={}$, {}'.format(length, init))
		# plt.plot(x, poly(x), '--', label=r'gradient={}'.format(poly.c[0]))
plt.legend()
plt.savefig('pattern.pdf')
plt.close()

print(2*np.pi/qc)
