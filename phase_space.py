import numpy as np
from StoEvolution import *
from FdEvolution import *
import matplotlib.pyplot as plt
import os


det_file = '../../NormalBox/Fdmethod/target_{}_u_{}'.format(-0.3, 1e-05)
sto_file = ['cycle_10', 'cycle_2', 'cycle_3', 'cycle_4']
path_to_slow_mfold = '../../NormalBox/slow_mfold'

figurename = 'phase_space.pdf'

plot_slow_mfold = True
plot_single_interface = True
plot_double_interface = False

phi_s = - np.sqrt(1/3)
phi_b = - 1


phi_bar_dot_evol = []
phi_bar_evol = []

for l in sto_file:
	solver = StoEvolution()
	solver.load(l)
	phi_bar_dot_evol = np.append(phi_bar_dot_evol,
						solver._average(-(solver.phi - solver.phi_target) * (solver.phi + solver.phi_shift)))
	phi_bar_evol = np.append(phi_bar_evol,
					solver._average(solver.phi))

	phi_shift = solver.phi_shift
	phi_target = solver.phi_target
	q_c = np.sqrt(solver.a/(2 * solver.k))
	X = solver.X

solver = FdEvolution()
solver.load(det_file)
det_phi_bar_dot = solver._average(-(solver.phi - phi_target) * (solver.phi + phi_shift))
det_phi_bar = solver._average(solver.phi)



if plot_slow_mfold:
	labels = []
	redundant = []

	for root, dirs, files in os.walk(path_to_slow_mfold):
		for f in files:
			if f.endswith('params.json'):
				label = f[:-12]
				labels.append(os.path.join(root, label))
				if label.endswith('cont'):
					redundant.append(os.path.join(root, label[:-5]))

	assert len(labels) > 0

	phi_bar_dot = []
	phi_bar = []
	for l in labels:
		if label not in redundant:
			solver = FdEvolution()
			solver.load(l)
			phi_bar_dot = np.append(phi_bar_dot,
							solver._average_vector(-(solver.phi[-1] - phi_target) * (solver.phi[-1] + phi_shift)))
			phi_bar = np.append(phi_bar,
						solver._average_vector(solver.phi[-1]))


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

plt.plot(det_phi_bar, det_phi_bar_dot, '-', label='deterministic')
plt.plot(phi_bar_evol, phi_bar_dot_evol, '-', label='stochastic')
plt.plot(phi_bar, phi_bar_dot, 'gx',  markersize=5, label='slow manifold')
plt.plot((min(phi_bar), max(det_phi_bar)), (0, 0), 'k--')


plt.xlabel(r'$\bar{\phi}$')
plt.ylabel(r'$\partial_t\bar{\phi}$')
plt.title(r'$\partial_t\bar{\phi}$ against $\phi$')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('{}.pdf'.format(figurename))
