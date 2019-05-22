import time
import numpy as np
from FdEvolution import *

rates = [3e-5, 1e-5, 3e-6, 1e-6, 3e-7]
Ls = [200, 300, 400]
n = np.zeros((len(Ls), 2, len(rates)))
for (i, u)in enumerate(rates):
	for (j, init) in enumerate(['flat', 'sin']):
		for (k, length) in enumerate(Ls):
			label = '{}_X_{}_u_{}'.format(init, length, u)
			print(label)
			solver = FdEvolution()
			solver.load(label)
			solver.rescale_to_standard()
			# solver.plot_steady_state(label, kink=0)
			# solver.plot_evolution(100, 100, label)
			n[k, j, i] = solver.count()
			print(n[k, j, i])

for (k, length) in enumerate(Ls):
	for (j, init) in enumerate(['sin']):
		plt.plot(np.log(rates), np.log(length/n[k, j]), 'x', label='{}_X_{}'.format(length, init))
plt.legend()
plt.savefig('pattern.pdf')
plt.close()
