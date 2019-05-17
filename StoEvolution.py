import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import ode
from scipy.fftpack import rfft, rfftfreq
from TimeEvolution import TimeEvolution
from FdEvolution import FdEvolution
import json


class StoEvolution(FdEvolution):

	def __init__(self, epsilon=None, a=None, k=None, u=None, phi_target=None, phi_shift=None):
		super().__init__(a, k, u, phi_target, phi_shift)
		self.epsilon = epsilon

	def initialise(self, X, dx, T, dt, n_batches, initial_value, flat=True):
		self.dx = dx
		self.size = int(X/dx)
		self.X = X
		self.T = T
		self.dt = dt
		self.n_batches = int(n_batches)
		self.step_size = T/(self.n_batches-1)
		self.batch_size = int(self.step_size/self.dt)
		self._modify_params()
		self.noise_matrix = self._make_noise_matrix()
		self.M1 = 1
		self.M2 = self.u*self.phi_shift**2*(2+self.phi_target*self.phi_shift)/2

		if flat:
			self.phi_initial = np.zeros((self.size))
		else:
			self.phi_initial = self._sin_surface(initial_value)

	def save_params(self, label):
		params = {
			'T': self.T,
			'dt': self.dt,
			'dx': self.dx,
			'X': self.X,
			'n_batches': self.n_batches,
			'step_size': self.step_size,
			'size': self.size,
			'epsilon': self.epsilon,
			'k': self.k,
			'u': self.u,
			'a': self.a,
			'phi_shift': self.phi_shift,
			'phi_target': self.phi_target,
		}

		with open('{}_params.json'.format(label), 'w') as f:
			json.dump(params, f)

	def load_params(self, label):
		with open('{}_params.json'.format(label), 'r') as f:
			params = json.load(f)
		self.epsilon = params['epsilon']
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

	def save_phi(self, label):
		np.save("{}_data.npy".format(label), self.phi)

	def load_phi(self, label):
		self.phi = np.load('{}_data.npy'.format(label))

	def save(self, label):
		self.save_params(label)
		self.save_phi(label)

	def load(self, label):
		self.load_params(label)
		self.load_phi(label)

	def evolve_euler(self):
		import sdeint

		F = lambda phi, t: self.delta(t, phi)
		G = lambda phi, t: self.noise_matrix
		tspan = np.arange(0, self.T, self.dt)
		self.phi = sdeint.itoEuler(F, G, self.phi_initial, tspan)
		return self.phi

	def evolve(self, verbose=True):
		self.noise_matrix = self._make_noise_matrix()

		self.phi = np.zeros((self.n_batches, self.size))
		r = ode(self._delta).set_integrator('lsoda', atol=1e-8)
		r.set_initial_value(self.phi_initial, 0)

		n = 0
		for i in range(int(self.T/self.dt)):
			if r.successful():
				if i % self.batch_size == 0:
					self.phi[n] = r.y
					if verbose:
						print('iteration: {}	mean: {}'.format(i, self._average_vector(r.y[2:-2])))
					n += 1
				r.set_initial_value(r.y + self._noisy_delta(), r.t)
				r.integrate(r.t+self.dt)


		return self.phi

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
		self.M = length_ratio**2 * self.M/time_ratio


	def rescale_to_standard(self):
		time_ratio = self.a**2/self.k
		space_ratio = np.sqrt(self.a/self.k)
		self.dx = space_ratio * self.dx
		self.X = space_ratio * self.X
		self.T = time_ratio * self.T
		self.dt = time_ratio * self.dt
		self.step_size = time_ratio * self.step_size
		self.a = 1
		self.k = 1
		self.u = self.u/time_ratio
		self.M = length_ratio**2 * self.M/time_ratio


	def _delta(self, t, phi):
		mu = self.a * (- phi + phi**3) - self.k * self._laplacian(phi)
		birth_death = - self.u * (phi + self.phi_shift) * (phi - self.phi_target)
		dphidt = self._laplacian(mu) + birth_death
		return self._enforce_bc(dphidt)

	def _make_noise_matrix(self):
		unit_matrix = np.identity(self.size)
		noise = np.roll(unit_matrix, 1, axis=-1) - np.roll(unit_matrix, -1, axis=-1)
		noise[:4, :] = 0
		noise[-4:, :] = 0
		noise[0, :] = noise[4, :]
		noise[-1, :] = noise[-5, :]
		noise[2, 3] = 2
		noise[-3, -4] = -2
		noise[1, 4] = 1
		noise[-2, -5] = -1
		noise[3, 4] = 1
		noise[-4, -5] = -1

		return np.sqrt(2 * self.M)/(2 * self.dx) * noise

	def _noisy_delta(self):
		dW = np.random.normal(0.0, np.sqrt(self.dt), (self.size))
		noise = np.matmul(self.noise_matrix, dW)
		return noise

	def _random_init(self, initial_value):
		phi_initial = initial_value + np.zeros((self.size))
		nsteps = int(self.a+10)
		r = ode(self._delta).set_integrator('lsoda', atol=1e-10, nsteps=nsteps)
		r.set_initial_value(phi_initial, 0)

		r.integrate(self.dt/2)
		r.set_initial_value(r.y + self._noisy_delta(), r.t)
		phi_initial = r.integrate(self.dt)
		return phi_initial

	def evolve_trajectories(self, n):
		# Evolve the profile forward n times
		self.phi_trajs = np.zeros((n, self.n_batches, self.size))
		for i in range(n):
			self.evolve(verbose=False)
			self.phi_trajs[i] = self.phi
			print('trajectory {} completed'.format(i))


	def save_trajs(self, label):
		np.save("{}_trajs.npy".format(label), self.phi_trajs)

	def load_trajs(self, label):
		self.phi_trajs = np.load('{}_trajs.npy'.format(label))
		self.phi = np.mean(self.phi_trajs, axis=0)

	def _compute_Fourier_components(self):
		# Extract the midpoints
		traj = (np.roll(self.phi_trajs, 1, axis=-1) + self.phi_trajs)/2
		traj = traj[:, :, 3:-2]

		# Extract the shape of phi and add 1 to the spatial axis
		x_size = int(traj.shape[-1]/2 + 1)
		shape = (traj.shape[0], traj.shape[1], x_size)

		# Fourier transform the midpoints
		phi_k = rfft(traj, axis=-1)
		S = np.zeros(shape)

		S[:, :, 0] = phi_k[:, :, 0] **2
		S[:, :, -1] = phi_k[:, :, -1]**2
		S[:, :, 1:-1] = (phi_k[:, :, 1:-2:2]**2 + phi_k[:, :, 2:-1:2]**2)

		S = np.mean(S, axis=0)
		return S


	def plot_fourier_components(self, label, truncate=True):
		# read in the structure factor
		S = self._compute_Fourier_components()

		# make axis for the third plot
		q_c = np.sqrt(self.a/self.k/2)
		t = np.linspace(0, self.T, self.n_batches)
		q = np.arange(0, 2*q_c, 2*np.pi/(self.X-self.dx))
		r = - self.u + self.a * q**2 - self.k * q**4

		# Truncate until we only have the first few compnents left
		if truncate:
			q_c = np.sqrt(self.a/self.k/2)
			q_size = int(2*q_c/(2*np.pi/(self.X-self.dx)))
			S = S[:, :q_size+1]

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		f, (ax1, ax2) = plt.subplots(2)

		for i in range(1, S.shape[-1]):
			ax1.plot(t, np.log(S[:, i]), label=r'$q_{}$'.format(i))
		ax1.legend()
		ax1.set_title(r'Average structure factor')
		ax1.set_xlabel(r't')
		ax1.set_ylabel(r'S')

		ax2.plot(q, r)
		ax2.set_ylim([-max(r), max(r)])
		ax2.set_xlabel(r'q')
		ax2.set_ylabel(r'rate')
		plt.tight_layout()
		plt.savefig('{}_sf.pdf'.format(label))
		plt.close()
