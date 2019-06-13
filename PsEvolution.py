import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as am
import time
from scipy.integrate import ode
from scipy.fftpack import fft2, ifft2, fftfreq
import pyfftw
from StoEvolutionPS import *


class PsEvolution(StoEvolutionPS):

	def evolve(self, verbose=True):
		self._make_k_grid()
		self._make_filters()
		self._set_up_fftw()
		self.phi = np.zeros((self.n_batches, self.size, self.size))
		phi = self.phi_initial

		small_batch = self.batch_size
		while small_batch > 10000:
			small_batch /= 10 # decrease the amount of time integration at each step

		r = ode(self._delta).set_integrator('vode', atol=1e-8, nsteps=small_batch)
		r.set_initial_value(self.phi_initial, 0)

		n = 0
		phi = self.phi_initial
		for i in range(int((self.T/self.dt)/small_batch)):
			if r.successful():
				if i % int(self.batch_size/small_batch) == 0:
					phi_complex = self._make_complex(phi)
					self.phi[n] = np.real(ifft2(phi_complex))
					if verbose:
						print('iteration: {}	mean: {}'.format(n, self._mean_bd(self.phi[n])))
					n += 1
				phi = r.integrate(r.t+self.dt*small_batch)

	def continue_evolution(self, T):
		self.phi_initial = self._make_real(fft2(self.phi[-2]))
		self.T = T
		self.n_batches = int(self.T/self.step_size+1)
		self.batch_size = int(self.step_size/self.dt)
		self.evolve()


	def _make_complex(self, phi):
		cutoff = int(self.size/2+1)
		phi = phi.reshape((self.size, cutoff, 2))
		phi_complex = np.empty((self.size, self.size), dtype=np.complex128)
		phi_complex[:, :cutoff] = phi[:, :, 0] + phi[:, :, 1]*1j
		phi_complex[:, cutoff:] = np.flip(phi_complex[:, 1:cutoff-1], axis=1).conj()
		return phi_complex

	def _make_real(self, phi_complex):
		cutoff = int(self.size/2+1)
		phi = phi_complex.view(np.float64)[:, :2*cutoff]
		return np.ravel(phi)

	def _delta(self, t, phi):
		phi_complex = self._make_complex(phi)
		dphidt_complex = super()._delta(phi_complex)
		return self._make_real(dphidt_complex)

	def _random_init(self, initial_value):
		self.input_forward[:] = np.random.normal(size=(self.size, self.size)).astype('complex128')
		dW = self.fft_forward()
		noise = self.dt*dW
		return self._make_real(noise)

	def _droplet_init(self, radius, skew):
		init = super()._droplet_init(radius, skew)
		return self._make_real(init)


if __name__ == '__main__':

	a = 0.2
	k = 1
	u = 3e-5
	phi_t = -0.7
	phi_shift = 10

	X = 128
	dx = 1
	T = 1e4
	dt = 1e-3
	n_batches = 100
	initial_value = -0.6
	flat = False

	label = 'phi_t_{}_l=2'.format(phi_t)

	start_time = time.time()
	solver = PsEvolution(0, a, k, u, phi_t, phi_shift)
	solver.initialise(X, dx, T, dt, n_batches, radius=18, skew=5, flat=flat)
	solver.evolve(verbose=True)
	solver.save(label)
	end_time = time.time()
	print('The simulation took: {}'.format(end_time - start_time))
