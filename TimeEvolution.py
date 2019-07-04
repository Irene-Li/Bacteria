import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import ode, cumtrapz
import json


class TimeEvolution:

	def __init__(self, a=None, k=None, u=None, phi_target=None, phi_shift=None):
		self.a = a
		self.k = k
		self.u = u
		self.phi_shift = phi_shift
		self.phi_target = phi_target

	def initialise(self, X, dx, T, dt, n_batches, initial_value, flat=False):
		self.dx = dx
		self.size = int(X/dx) #+ 5
		self.X = X
		self.T = T
		self.dt = dt
		self.n_batches = int(n_batches)
		self.step_size = T/(self.n_batches-1)
		self.batch_size = int(np.floor(self.step_size/self.dt))
		self._modify_params()

		if flat:
			self.phi_initial = self._random_init(initial_value)
		else:
			self.phi_initial = self._sin_surface(initial_value)

	def _modify_params(self):
		length_ratio = 1/self.dx
		self.dx = 1
		self.X = self.X * length_ratio
		self.a = (1/self.k)*self.a/length_ratio**2
		time_ratio = length_ratio**4 * (self.k/1)
		self.k = 1
		self.T = self.T * time_ratio
		self.u = self.u/time_ratio
		self.dt = self.dt * time_ratio
		self.step_size = self.step_size * time_ratio

	def save(self, label):
		np.save("{}_data.npy".format(label), self.phi)

		params = {
			'T': self.T,
			'dt': self.dt,
			'dx': self.dx,
			'X': self.X,
			'n_batches': self.n_batches,
			'step_size': self.step_size,
			'size': self.size,
			'a': self.a,
			'k': self.k,
			'u': self.u,
			'phi_shift': self.phi_shift,
			'phi_target': self.phi_target
		}

		with open('{}_params.json'.format(label), 'w') as f:
			json.dump(params, f)


	def load(self, label):
		self.phi = np.load('{}_data.npy'.format(label))

		with open('{}_params.json'.format(label), 'r') as f:
			params = json.load(f)
		self.a = params['a']
		self.k = params['k']
		self.u = params['u']
		self.phi_shift = params['phi_shift']
		self.phi_target = params['phi_target']
		self.X = params['X']
		self.T = params['T']
		self.dt = params['dt']
		self.dx = params['dx']
		self.size = params['size']
		self.n_batches = params['n_batches']
		self.step_size = params['step_size']

	def evolve(self):
		pass


	def continue_evolution(self, n_steps):
		self.phi_initial = self.phi[-2]
		self.T = self.dt * n_steps
		self.batch_size = int(self.step_size/self.dt)
		self.n_batches = int(n_steps/self.batch_size)
		self.evolve()

	def count(self):
		phi_final = self.phi[-2]
		bool_array = (phi_final>self.phi_target)
		bool_array = np.abs(bool_array - np.roll(bool_array, -1))
		return np.sum(bool_array)/2


	def plot_evolution(self, label, t_size=100, x_size=100):
		(t_size_old, x_size_old) = self.phi.shape
		t_grid_size = int(np.ceil(t_size_old/t_size))
		x_grid_size = int(np.ceil(x_size_old/x_size))
		phi_plot = self.phi[::t_grid_size, ::x_grid_size]

		T = t_grid_size * self.step_size * phi_plot.shape[0]
		X = x_grid_size * self.dx * phi_plot.shape[1]
		ymesh, xmesh = np.mgrid[slice(0, T, self.step_size*t_grid_size),
								slice(0, X, self.dx*x_grid_size)]

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif', size=15)

		plt.figure(figsize=(10, 10))
		plt.pcolor(xmesh, ymesh, phi_plot, cmap='plasma', edgecolors='face', alpha=1)
		plt.colorbar()
		plt.xlabel(r'x')
		plt.ylabel(r't')
		plt.title(r'Spacetime plot')
		plt.tight_layout()
		plt.savefig('{}_evolution.pdf'.format(label))
		plt.close()

	def plot_free_energy(self, label):
		f = self._compute_f()
		t = np.linspace(0, self.T, self.n_batches)
		phi_bar = self._average(self.phi)
		f_flat = - self.a/2 * phi_bar**2 + self.a/4 * phi_bar**4
		q_c = np.sqrt(self.a/(2 * self.k))
		f_tanh =  - 1/4 * self.a + 1/(3 * q_c * self.X) * self.a + self.k*2*q_c/(3 * self.X)
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')


		plt.plot(t, f, 'k-', linewidth=2, label='True free energy')
		plt.plot(t, f_flat, 'y--', linewidth=1.5, label='Free energy for a uniform distribution')
		plt.plot((t[0], t[-1]), (f_tanh, f_tanh), 'c--', linewidth=1.5, label='Free energy using tanh approximation')
		plt.title(r'$F[\phi]$ over time')
		plt.xlabel(r't')
		plt.ylabel(r'Free energy density')
		plt.legend()
		plt.tight_layout()
		plt.savefig('{}_free_energy.pdf'.format(label))
		plt.close()

	def plot_current(self, label):
		phi = self.phi[-1, 2:-2]
		phi_dot = - (phi- self.phi_target) * (phi + self.phi_shift)
		x = np.arange(0, (self.size)* self.dx, self.dx)[2:-2]
		current = - cumtrapz(phi_dot, x, initial=0)

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		plt.plot(x, current)
		plt.title(r'$J(x)$ in steady state')
		plt.xlabel(r'$x$')
		plt.ylabel(r'$J(x)$')
		plt.tight_layout()
		plt.savefig('{}_current.pdf'.format(label))
		plt.close()

	def plot_phi_bar_dot(self, label):
		phi_dot = -(self.phi- self.phi_target) * (self.phi + self.phi_shift)
		phi_bar_dot = self._average(phi_dot)
		print(phi_bar_dot[-1])

		t = np.linspace(0, self.T, self.n_batches)

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		plt.plot(t, phi_bar_dot, 'b-')
		plt.title(r'$\partial_t\bar{\phi}$ over time')
		plt.xlabel(r't')
		plt.ylabel(r'phi_bar_dot')
		plt.tight_layout()
		plt.savefig('{}_phi_bar_dot.pdf'.format(label))
		plt.close()


	def plot_phase_space(self, label):
		phi_bar = self._average(self.phi)
		phi_dot = -(self.phi- self.phi_target) * (self.phi + self.phi_shift)
		phi_bar_dot = self._average(phi_dot)

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		plt.plot(phi_bar, phi_bar_dot, 'k-')
		plt.xlabel(r'$\bar{\phi}$')
		plt.ylabel(r'$\partial_t\bar{\phi}$')
		plt.title(r'Phase space plot')
		plt.tight_layout()
		plt.savefig('{}_phase_space.pdf'.format(label))
		plt.close()


	def plot_average(self, label):
		self.phi_average = self._average(self.phi)
		t = np.linspace(0, self.T, self.n_batches)
		phi_b = 1
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		plt.plot(t, self.phi_average, 'b-')
		# plt.plot((0, self.T), (-self.phi_target, self.phi_target), 'k--')
		# plt.plot((0, self.T), (-phi_b, -phi_b), 'g--')
		plt.title(r'Evolution of $\bar{\phi}$ over time')
		plt.xlabel(r't')
		plt.ylabel(r'$\bar{\phi}$')
		plt.savefig('{}_average.pdf'.format(label))
		plt.close()

	def plot_samples(self, label, n=2):
		assert n >= 0
		phi_b = 1
		x = np.arange(0, (self.size)* self.dx, self.dx)
		step = int(self.n_batches/(n-1)-1)

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		f, (ax1, ax2) = plt.subplots(2, sharex=True)
		for i in range(0, self.n_batches, step):
			ax1.plot(x[2:-2], self.phi[i, 2:-2], '-', label=r't ={}'.format(i*self.step_size))
			phi_dot = - (self.phi[i, 2:-2] + self.phi_shift) * (self.phi[i, 2:-2] - self.phi_target)
			ax2.plot(x[2:-2], phi_dot, '-', label=r't ={}'.format(i*self.step_size))

		ax1.plot((0, x[-1]), (self.phi_target, self.phi_target), 'y--')
		ax1.set_title(r'Samples of $\phi$ over time')

		f.subplots_adjust(hspace=0)
		plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

		ax2.set_xlabel(r'$x$')
		ax1.set_ylabel(r'$\phi$')
		ax2.set_ylabel(r'$\partial_t\phi$ due to birth-death')
		ax1.legend(loc='upper left')
		ax2.legend(loc='lower left')
		plt.savefig('{}_final.pdf'.format(label))
		plt.close()

	def plot_steady_state(self, label):
		x = np.arange(0, (self.size)* self.dx, self.dx)

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.plot(x, self.phi[-2])
		plt.axhline(x=self.phi_target, color=k)
		plt.xlabel(r'$x$')
		plt.ylabel(r'$\phi$')
		plt.xlim([0, self.size*self.dx])
		plt.ylim([-1, 1])
		plt.savefig('{}_final.pdf'.format(label))
		plt.close()

	def rescale_to_standard(self):
		pass

	def _sin_surface(self, phi_average):
		x = np.arange(0, self.dx * (self.size), self.dx)
		phi_initial = phi_average + 0.6*np.cos(2*np.pi*x/self.X)
		return self._enforce_bc(phi_initial)

	def _make_shifted_interface(self, phi_average):
		shift = 0.1
		new_phi_average = phi_average + shift
		self.phi_initial = self._slow_mfold(new_phi_average) - shift
		self.phi_initial = self._enforce_bc(self.phi_initial)

	def _random_init(self, phi_average):
		noise_amplitude = self.a * self.k * self.dx**4/1e6
		phi_initial = phi_average + noise_amplitude * np.random.normal(size=self.size)
		phi_initial = self._enforce_bc(phi_initial)
		return phi_initial



	def _slow_mfold(self, phi_average):
		q_c = np.sqrt(self.a/(2 * self.k))
		phi_b = 1
		x_0 = (- phi_average/(2 * phi_b) + 0.5) * self.X

		x = np.arange(0, self.size * self.dx, self.dx)
		phi_initial = phi_b * np.tanh(q_c * (x - x_0))

		return self._enforce_bc(phi_initial)

	def _double_tanh(self, phi_average):
		q_c = np.sqrt(self.a/(2 * self.k))

		x_1 = (phi_average/4 + 0.3) * self.X
		x_2 = (- phi_average/4 + 0.6) * self.X

		x = np.arange(0, self.size * self.dx, self.dx)
		phi_initial = 1 - np.tanh(q_c*(x - x_1))+ np.tanh(q_c*(x - x_2))
		return self._enforce_bc(phi_initial)


	def _average(self, phi):
		l = phi.shape[-1]
		s = np.sum(phi[:, 1:-1], axis = -1)
		s += 0.5 * (phi[:, 0] + phi[:, -1])
		return s/(l-1)

	def _average_vector(self, phi):
		l = phi.size
		s = np.sum(phi[1:-1])
		s += 0.5 * (phi[0] + phi[-1])
		return s/(l-1)

	def _mean_bd(self, phi):
		bd = - (phi + self.phi_shift)*(phi - self.phi_target)
		return np.sum(bd)
