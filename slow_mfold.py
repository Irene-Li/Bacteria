import numpy as np
from FdEvolution import *
import matplotlib.pyplot as plt
import os

phi_target = -0.35
u = 5e-6

# figurename = 'slow_mfold_{}_{}'.format(target, u)
label = 'phi_t_{}_u_{}'.format(phi_target, u)


figurename = 'slow_mfold'

plot_slow_mfold = True
plot_spinodal = False
plot_single_interface = False
plot_double_interface = True

phi_s = - np.sqrt(1/(3))
phi_b = - 1

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)


# Read in the evolution phase space

solver = FdEvolution()
solver.load(label)

phi = solver.phi[25:-1]
phi_bar_dot_evol = np.mean(-(phi - solver.phi_target) * (phi + solver.phi_shift), axis=1)
phi_bar_evol = np.mean(phi, axis=-1)

phi_shift = solver.phi_shift
phi_target = solver.phi_target
q_c = np.sqrt(solver.a/(2 * solver.k))
X = solver.size * solver.dx

max_phi_bar = max(phi_bar_evol)
min_phi_bar = min(phi_bar_evol)
max_phi_bar_dot = max(phi_bar_dot_evol)
min_phi_bar_dot = min(phi_bar_dot_evol)


if plot_slow_mfold:
	labels = []
	redundant = []
	directory = 'slow_mfold'

	for root, dirs, files in os.walk(directory):
		for f in files:
			if f.endswith('params.json'):
				label = f[:-12]
				labels.append(os.path.join(root, label))

	assert len(labels) > 0

	phi_bar_dot = []
	phi_bar = []
	solver = FdEvolution()
	solver.load(labels[0])

	for label in labels:
		solver.load(label)
		phi = solver.phi[-2]
		phi_bar_dot.append(np.mean(-(phi - phi_target) * (phi + phi_shift)))
		phi_bar.append(np.mean(phi))

	max_phi_bar = max(max_phi_bar, max(phi_bar))+0.01
	min_phi_bar = min(min_phi_bar, min(phi_bar))-0.01
	max_phi_bar_dot = max(max_phi_bar_dot, max(phi_bar_dot))
	min_phi_bar_dot = min(min_phi_bar_dot, min(phi_bar_dot))

x = np.linspace(min_phi_bar, max_phi_bar, 100)
plt.plot((min_phi_bar, max_phi_bar), (0, 0), 'k--')
plt.plot(x, -(x - phi_target) * (x + phi_shift), 'y--', label='uniform')
if plot_spinodal:
	plt.plot((phi_s, phi_s), (min_phi_bar_dot, max_phi_bar_dot), 'r--', label='spinodal density')
if plot_single_interface:
	phi_new_target = (phi_shift * phi_target - phi_b ** 2 * (1 - 2/(q_c * X)))/(phi_shift - phi_target)
	plt.plot(x, - (phi_shift - phi_target) * (x - phi_new_target), '--', label='tanh')
if plot_double_interface:
	phi_new_target = (phi_shift * phi_target - phi_b ** 2 * (1 - 4/(q_c * X)))/(phi_shift - phi_target)
	plt.plot(x, - (phi_shift - phi_target) * (x - phi_new_target), '--', label='double interface')
if plot_slow_mfold:
	plt.plot(phi_bar, phi_bar_dot, 'gx', markersize=5, label='slow manifold')

plt.plot(phi_bar_evol, phi_bar_dot_evol, 'k-', label='time evolution')



plt.xlabel(r'$\bar{\phi}$')
plt.ylabel(r'$\partial_t\bar{\phi}$')
plt.xlim([min_phi_bar, max_phi_bar])
# plt.ylim([-1, +0.5])
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('{}.pdf'.format(figurename))
