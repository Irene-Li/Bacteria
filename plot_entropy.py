import numpy as np
import matplotlib.pyplot as plt
from EvolveModelAB import *

M2 = 5e-6
delta_bs = [0, 0.03, 0.06, 0.1, 0.13, 0.16, 0.2, 0.23, 0.26, 0.3, 0.32, 0.36, 0.4, 0.46, 0.5]
# delta_bs = [0.1]
entropies_phi = np.empty(len(delta_bs), dtype='float')

for (i, delta_b) in enumerate(delta_bs):

    label = 'M2_{}_delta_b_{}'.format(M2, delta_b)
    print(label)

    solver = EntropyModelAB()
    solver.load(label)
    solver.read_entropy(label+'_phi')
    solver.plot_entropy(label+'_phi')
    entropies_phi[i] = np.real(np.sum(solver.entropy))

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)

x = np.array(delta_bs)
y = np.real(entropies_phi)
logy = np.log(np.real(entropies_phi)[x>0])
logx = np.log(x[x>0])

z = np.polyfit(logx[:10], logy[:10], 1)
p = np.poly1d(z)
grad = z[0]
print(grad)

plt.plot(x, y, '+', markersize=8, markeredgewidth=1.5)
plt.yticks([0])
plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
plt.xlim([0, 0.51])
plt.ylim([np.min(y), 1.05*np.max(y)])
plt.xlabel(r'$\beta^\prime - \beta$')
plt.ylabel(r'$\dot{S}_\mathrm{ss}[\phi]$')

plt.axes([0.2, 0.45, 0.2, 0.4], facecolor='w', alpha=0.3)
plt.plot(logx, logy, '+', markersize=5, color='darkorange')
plt.xticks([])
plt.yticks([])
plt.title('\Large{log-log plot}')

# plot a triangle
origin = np.array([-2.5, -17.6])
dx = np.array([1, 0])
dy = np.array([0, 2])
x_label = origin + dx/2 - 0.2*dy
y_label = origin + dx + dy/3 + 0.1*dx

triangle = plt.Polygon([origin, origin+dx, origin+dx+dy], alpha=0.3)
plt.gca().add_patch(triangle)
plt.text(x_label[0], x_label[1], "\large{1}")
plt.text(y_label[0], y_label[1], "\large{2}")


plt.tight_layout()
# plt.show()
plt.savefig('total_EPR_phi.pdf')
plt.close()
