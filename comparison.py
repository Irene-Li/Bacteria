from FE import *
from PDE import * 
from matplotlib import pyplot as plt 
import numpy as np 

dxs = ['01', '05']
dxs_n = [0.1, 0.5]
skips = [5, 1]

dts = ['02', '05', '1', '5', '10', '20']
dts_n = [0.2, 0.5, 1, 5, 10, 20]

l1 = len(dxs)
l2 = len(dts) 

phi = np.zeros((l1, l2, 121))
for i in range(l1):
	for j in range(l2):
		label = 'dx_{}_dt_{}'.format(dxs[i], dts[j])
		solver = FeEvolution()
		solver.load(label)
		phi[i, j] = solver.phi[-1, ::skips[i]]

compare_to = phi[0, 0, :]
diff = np.log(np.mean(np.abs(phi - compare_to), axis=-1))

for i in range(l1):
	plt.plot(np.log(dts_n), diff[i], 'o', label='dx = {}'.format(dxs_n[i]))
plt.legend()
plt.title('error plot')
plt.ylabel('mean error')
plt.xlabel('log(dt)')
plt.show()
plt.savefig('comparison.pdf')


