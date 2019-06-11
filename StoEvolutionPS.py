import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as am
import time
import scipy.sparse as sp
from scipy.fftpack import fft2, ifft2, fftfreq
import pyfftw
import json
from StoEvolution import *


class StoEvolutionPS(StoEvolution):

	def evolve(self, verbose=True):
		self._make_k_grid()
		self._make_filters()
		self._set_up_fftw()
		self.phi = np.zeros((self.n_batches, self.size, self.size))
		phi = self.phi_initial

		n = 0
		for i in range(int(self.T/self.dt)):
			if i % self.batch_size == 0:
				self.phi[n] = np.real(ifft2(phi))
				if verbose:
					print('iteration: {}	mean bd: {}'.format(n, self._mean_bd(self.phi[n])))
				n += 1
			phi += self._delta(phi)*self.dt + self._noisy_delta()
	def initialise(self, X, dx, T, dt, n_batches, initial_value=0, radius=20, skew=0, flat=True):
		super().initialise(X, dx, T, dt, n_batches, initial_value, flat=True)
		if not flat:
			self.phi_initial = self._droplet_init(radius, skew)

	def continue_evolution(self, T):
		self.phi_initial = fft2(self.phi[-2])
		self.T = T
		self.n_batches = int(self.T/self.step_size+1)
		self.batch_size = int(self.step_size/self.dt)
		self.evolve()

	def double_droplet_init(self):
		x = np.arange(self.size)
		y = np.arange(self.size)
		x, y = np.meshgrid(x, y)
		midpoint1 = int(self.size/3)
		midpoint2 = 2*midpoint1
		l = np.sqrt(self.k/self.a)
		size1 = int(self.size/6)
		size2 = int(self.size/10)

		phi1 = - np.tanh((np.sqrt((x-midpoint1)**2+(y-midpoint1)**2)-size1)/l)+1
		phi2 = - np.tanh((np.sqrt((x-midpoint2)**2+(y-midpoint2)**2)-size2)/l)+1
		phi = phi1+phi2-1
		self.phi_initial = fft2(phi)

	def _plot_state(self, phi, n):
		plt.imshow(phi)
		plt.colorbar()
		plt.savefig('state_{}.pdf'.format(n))
		plt.close()

	def _make_k_grid(self):
		Nx, Ny = self.size, self.size
		kx = fftfreq(Nx)*2*np.pi
		ky = fftfreq(Ny)*2*np.pi
		self.kx, self.ky = np.meshgrid(kx, ky)
		self.ksq = self.kx*self.kx + self.ky*self.ky

	def _make_filters(self):
		kmax = np.max(np.abs(self.kx))
		filtr = (np.abs(self.kx) > kmax*2/3)
		filtr2 = (np.abs(self.kx) > kmax*1/2)

		self.dealiasing_double = filtr | filtr.T
		self.dealiasing_triple = filtr2 | filtr2.T

	def _set_up_fftw(self):
		self.input_forward = pyfftw.empty_aligned((self.size, self.size), dtype='complex128')
		output_forward = pyfftw.empty_aligned((self.size, self.size), dtype='complex128')
		self.fft_forward = pyfftw.FFTW(self.input_forward, output_forward,
										direction='FFTW_FORWARD', axes=(0, 1),
										flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])
		self.input_backward = pyfftw.empty_aligned((self.size, self.size), dtype='complex128')
		output_backward = pyfftw.empty_aligned((self.size, self.size), dtype='complex128')
		self.fft_backward = pyfftw.FFTW(self.input_backward, output_backward,
										direction='FFTW_BACKWARD', axes=(0, 1),
										flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])

	def _delta(self, phi):
		self.input_backward[:] = phi
		phi_x = self.fft_backward()
		self.input_forward[:] = phi_x*phi_x
		phi_sq = self.fft_forward()
		self.input_forward[:] = phi_x**3
		phi_cube = self.fft_forward()
		np.putmask(phi_cube, self.dealiasing_triple, 0)
		np.putmask(phi_sq, self.dealiasing_double, 0)

		mu = self.a*(-phi+phi_cube) + self.k*self.ksq*phi
		birth_death = - self.u*(phi_sq+(self.phi_shift-self.phi_target)*phi)
		birth_death[0, 0] += self.u*self.phi_shift*self.phi_target*self.size**2
		dphidt = -self.M1*self.ksq*mu + birth_death
		return dphidt

	def _noisy_delta(self):
		self.input_forward[:] = np.random.normal(size=(self.size, self.size)).astype('complex128')
		dW = self.fft_forward()
		noise = np.sqrt(2*(self.M2+self.ksq*self.M1)*self.epsilon*self.dt)*dW
		return noise

	def _flat_surface(self, initial_value):
		phi = np.zeros((self.size, self.size))+0j
		phi[0, 0] += initial_value*(self.size)**2
		return phi

	def _droplet_init(self, radius, skew):
		x = np.arange(self.size)
		y = np.arange(self.size)
		x, y = np.meshgrid(x, y)
		midpoint = int(self.size/2)
		l = np.sqrt(self.k/self.a)
		x_skew = (1-skew)**2
		y_skew = (1+skew)**2
		phi = 0.7*(- np.tanh((np.sqrt(x_skew*(x-midpoint)**2+y_skew*(y-midpoint)**2)-radius)/l)+1)
		phi += self.phi_target
		return fft2(phi)

	def plot_slices(self, label):
		phi = self.phi[-1]
		for n in range(0, self.size, 20):
			plt.plot(phi[n])
		plt.show()


	def make_movie(self, label, t_grid=1):
		fig = plt.figure()
		low, high = -1.2, 1.2
		ims = []
		im = plt.imshow(self.phi[0], vmin=low, vmax=high, animated=True)
		plt.colorbar(im)
		for i in range(self.n_batches):
			xy = self.phi[i]
			im = plt.imshow(xy, vmin=low, vmax=high, animated=True, cmap='seismic')
			ims.append([im])
		ani = am.ArtistAnimation(fig, ims, interval=100, blit=True,
										repeat_delay=1000)
		mywriter = am.FFMpegWriter()
		ani.save(label+"_movie.mp4", writer=mywriter)
		plt.close()

	def make_curve_movie(self, label):
		fig = plt.figure()
		phi = (self.phi > 0.5).astype('float64')
		ims = []
		for i in range(self.n_batches):
			xy = phi[i]
			im = plt.imshow(xy, animated=True)
			ims.append([im])
		plt.colorbar(im)
		ani = am.ArtistAnimation(fig, ims, interval=100, blit=True,
										repeat_delay=1000)
		mywriter = am.FFMpegWriter()
		ani.save(label+"_curve_movie.mp4", writer=mywriter)
		plt.close()



if __name__ == '__main__':

	epsilon = 0.1
	a = 0.2
	k = 1
	u = 1e-5
	phi_t = 0
	phi_shift = 10

	X = 128
	dx = 1
	T = 100
	dt = 5e-3
	n_batches = 100
	initial_value = 0
	flat = False

	for u in [5e-5]:
		label = 'u_{}_test'.format(u)
		initial_value = phi_t

		start_time = time.time()
		solver = StoEvolutionPS(epsilon, a, k, u, phi_t, phi_shift)
		solver.initialise(X, dx, T, dt, n_batches, initial_value, flat=flat)
		solver.save_params(label)
		solver.print_params()
		solver.evolve(verbose=False)
		solver.save_phi(label)
		end_time = time.time()
		print('The simulation took: {}'.format(end_time - start_time))
