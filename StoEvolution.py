import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import ode
from scipy.fftpack import irfft, rfft, fft
from TimeEvolution import TimeEvolution
from FdEvolution import FdEvolution
import json
import scipy.sparse as sp


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
		self.M1 = 1
		self.M2 = self.u*(self.phi_shift+self.phi_target/2)
		self._modify_params()
		self._make_gradient_matrix()

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

	def evolve(self, verbose=True):
		self.phi = np.zeros((self.n_batches, self.size))
		self._make_laplacian_matrix()
		r = ode(self._delta).set_integrator('lsoda', atol=1e-8)
		r.set_initial_value(self.phi_initial, 0)

		n = 0
		for i in range(int(self.T/self.dt)):
			if r.successful():
				if i % self.batch_size == 0:
					self.phi[n] = r.y
					if verbose:
						print('iteration: {}	mean: {}'.format(i, np.mean(r.y)))
					n += 1
				r.set_initial_value(r.y + self._noisy_delta(), r.t)
				r.integrate(r.t+self.dt)
		return self.phi

	def _modify_params(self):
		length_ratio = self.dx
		time_ratio = length_ratio**4/self.k #such that M1*k=1

		self.dx = 1
		self.X /= length_ratio
		self.T /= time_ratio
		self.dt /= time_ratio
		self.step_size /= time_ratio

		self.k /= length_ratio**2
		self.M1 *= time_ratio/length_ratio**2
		self.u *= time_ratio
		self.M2 *= time_ratio
		self.epsilon /= length_ratio

	def print_params(self):
		print('X', self.X, '\n',
		'dx', self.dx, '\n',
		'T', self.T, '\n',
		'dt', self.dt, '\n',
		'M1', self.M1, '\n',
		'M2', self.M2, '\n',
		'a', self.a, '\n',
		'k', self.k, '\n',
		'u', self.u, '\n',
		'epsilon', self.epsilon, '\n'
		)

	def rescale_to_standard(self):
		pass


	def _delta(self, t, phi):
		mu = self.a * (- phi + phi**3) - self.k * self._laplacian(phi)
		birth_death = - self.u * (phi + self.phi_shift) * (phi - self.phi_target)
		dphidt = self.M1 * self._laplacian(mu) + birth_death
		return dphidt

	def _make_gradient_matrix(self):
		n = int(self.size/2)
		x = np.arange(n)
		laplacian_fourier = - 2 * (1 - np.cos(2 * np.pi * x/self.size))
		self._gradient_fourier = np.sqrt(- laplacian_fourier)

	def _noisy_delta(self):
		# noise of the conservative dynamics
		dW = np.random.normal(0.0, np.sqrt(self.epsilon*self.dt*self.size), self.size)
		dW[0] *= np.sqrt(2)
		dW[-1] *= np.sqrt(2)
		noise_fourier = np.sqrt(self.M2)*dW
		noise_fourier[1:self.size-1:2] -= np.sqrt(self.M1)*self._gradient_fourier[1:]*dW[2:self.size-1:2]
		noise_fourier[2:self.size-1:2] += np.sqrt(self.M1)*self._gradient_fourier[1:]*dW[1:self.size-1:2]
		return irfft(noise_fourier)

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
