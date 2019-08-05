import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import ode
from scipy.fftpack import irfft, rfft, fft
from TimeEvolution import TimeEvolution
from DetEvolution1D import DetEvolution1D
import json
import scipy.sparse as sp
import os
from pseudospectral import evolve_sto_ps_1d


class StoEvolution1D(DetEvolution1D):

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

		if flat:
			self.phi_initial = self._flat_surface(initial_value)
		else:
			self.phi_initial = self._double_tanh(initial_value)

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
		self.M1 = 1/self.k
		self.M2 = self.u*(self.phi_shift+self.phi_target/2)

	def save_phi(self, label):
		filename = "{}_data.npy".format(label)
		np.save(filename, self.phi)

	def load_phi(self, label):
		filename = "{}_data.npy".format(label)
		self.phi = np.load(filename)

	def save(self, label):
		self.save_params(label)
		self.save_phi(label)

	def load(self, label):
		self.load_params(label)
		self.load_phi(label)

	def evolve(self, verbose=True, ps=True):
		if ps:
			self.evolve_ps()
		else:
			self.evolve_fd(verbose)

	def evolve_ps(self):
		phi = fft(self.phi_initial)
		nitr = int(self.T/self.dt)
		self.phi = evolve_sto_ps_1d(phi, self.a, self.k, self.u,
									self.phi_shift, self.phi_target, self.epsilon,
									self.dt, nitr, self.n_batches, self.X)


	def evolve_fd(self, verbose):
		self.phi = np.zeros((self.n_batches, self.size))
		self._make_laplacian_matrix()
		self._make_laplacian_fourier()
		phi = self.phi_initial
		n = 0
		for i in range(int(self.T/self.dt)):
			if i % self.batch_size == 0:
				self.phi[n] = phi
				if verbose:
					print('iteration: {}	mean: {}'.format(n, np.mean(phi)))
				n += 1
			phi += self._delta(phi)*self.dt+self._noisy_delta()

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
		'epsilon', self.epsilon, '\n',
		'phi_shift', self.phi_shift, '\n',
		'phi_target', self.phi_target, '\n'
		)

	def rescale_to_standard(self):
		pass

	def _flat_surface(self, initial_value):
		return np.zeros((self.size)) + initial_value

	def _delta(self, phi):
		mu = self.a * (- phi + phi**3) - self.k * self._laplacian(phi)
		birth_death = - self.u * (phi + self.phi_shift) * (phi - self.phi_target)
		dphidt = self.M1 * self._laplacian(mu) + birth_death
		return dphidt

	def _make_laplacian_fourier(self):
		n = int(self.size/2+1)
		x = np.arange(n)
		self._laplacian_fourier = - 2 * (1 - np.cos(2 * np.pi * x/self.size))

	def _noisy_delta(self):
		# noise of the conservative dynamics
		dW = np.random.normal(0.0, np.sqrt(self.epsilon*self.dt*self.size), self.size)
		dW[0]*=np.sqrt(2)
		dW[-1]*=np.sqrt(2)
		noise_fourier = np.empty((self.size), dtype=np.float64)
		noise_fourier[0] = dW[0]*np.sqrt(self.M2)
		noise_fourier[1::2] = np.sqrt(-self.M1*self._laplacian_fourier[1:] + self.M2)*dW[1::2]
		noise_fourier[2:self.size-1:2] = np.sqrt(-self.M1*self._laplacian_fourier[1:-1] + self.M2)*dW[2:self.size-1:2]
		return irfft(noise_fourier)

	def _random_init(self, initial_value):
		phi_initial = initial_value + np.zeros((self.size))
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
