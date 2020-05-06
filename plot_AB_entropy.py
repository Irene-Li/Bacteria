import numpy as np
import matplotlib.pyplot as plt
from EvolveModelAB import *

M2 = 5e-6
delta_bs = [0, 0.03, 0.06, 0.1, 0.13, 0.16, 0.2, 0.23, 0.26, 0.3, 0.32, 0.36, 0.4, 0.46, 0.5]
# delta_bs = [0.1]
entropies_current = np.empty(len(delta_bs), dtype='float')


for (i, delta_b) in enumerate(delta_bs):

    label = 'M2_{}_delta_b_{}'.format(M2, delta_b)
    print(label)

    solver = EntropyModelAB()
    solver.load(label)
    solver.read_entropy(label+'_current')
    # solver.calculate_entropy_ratio()
    # solver.plot_entropy(label+'_current', current=True)
    entropies_current[i] = np.real(np.sum(solver.entropy))

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)

x = np.array(delta_bs)
y = np.real(entropies_current)

plt.plot(x, y, '+', markersize=8, markeredgewidth=1.5)
plt.yticks([0])
plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
plt.xlim([0, 0.51])
plt.ylim([0, 1.05*np.max(y)])
plt.xlabel(r'$\beta^\prime - \beta$')
plt.ylabel(r'$\epsilon \dot{S}^\mathrm{AB}_\mathrm{ss}[\phi]$')

plt.tight_layout()
# plt.show()
plt.savefig('total_EPR_phiAB.pdf')
plt.close()
