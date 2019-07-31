import time
import numpy as np
from FdEvolution import *

deltas = [1000, 3000, 1e4]
Ls = [400]
n = np.zeros((len(Ls), len(deltas)))
for (i, d)in enumerate(deltas):
	for (k, length) in enumerate(Ls):
		label = 'X_{}_delta_{}_large'.format(length, d)
		print(label)
		solver = FdEvolution()
		solver.load(label)
		solver.rescale_to_standard()
		# solver.plot_steady_state(label)
		solver.plot_evolution(label, 100, 100)
		n[k, i] = solver.count()
		print(n[k, i])

for (k, length) in enumerate(Ls):
	plt.plot(np.log(deltas), np.log(length/n[k]), 'x', label='X={}'.format(length))
plt.legend()
plt.savefig('pattern.pdf')
plt.close()
