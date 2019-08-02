import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as am
import time
import scipy.sparse as sp
from scipy.fftpack import fftfreq
import json
from StoEvolution import *
from pseudospectral import evolve_sto_ps
import mkl_fft


class StoEvolutionPS(StoEvolution):

	def evolve(self, verbose=True, cython=True):
		if cython:
			nitr = int(self.T/self.dt)
			self.phi = evolve_sto_ps(self.phi_initial, self.a, self.k, self.u,
										self.phi_shift, self.phi_target, self.epsilon,
										self.dt, nitr, self.n_batches, self.X)
		else:
			self.naive_evolve(verbose)

	def naive_evolve(self, verbose):
		self._make_k_grid()
		self._make_filters()
		self.phi = np.zeros((self.n_batches, self.size, self.size))
		phi = self.phi_initial

		n = 0
		for i in range(int(self.T/self.dt)):
			if i % self.batch_size == 0:
				self.phi[n] = np.real(mkl_fft.ifft2(phi))
				if verbose:
					print('iteration: {}	mean bd: {}'.format(n, self._mean_bd(self.phi[n])))
				n += 1
			phi += self._delta(phi)*self.dt + self._noisy_delta()


	def initialise(self, X, dx, T, dt, n_batches, initial_value=0, radius=20, skew=0, flat=True):
		super().initialise(X, dx, T, dt, n_batches, initial_value, flat=True)
		if not flat:
			self.phi_initial = self._droplet_init(radius, skew)

	def continue_evolution(self, T):
		self.phi_initial = mkl_fft.fft2(self.phi[-2])
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
		self.phi_initial = mkl_fft.fft2(phi)

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

	def _delta(self, phi):
		phi_x = mkl_fft.ifft2(phi)
		self.phi_sq = mkl_fft.fft2(phi_x**2)
		self.phi_cube = mkl_fft.fft2(phi_x**3)
		np.putmask(self.phi_cube, self.dealiasing_triple, 0)
		np.putmask(self.phi_sq, self.dealiasing_double, 0)

		mu = self.a*(-phi+self.phi_cube) + self.k*self.ksq*phi
		birth_death = - self.u*(self.phi_sq+(self.phi_shift-self.phi_target)*phi)
		birth_death[0, 0] += self.u*self.phi_shift*self.phi_target*self.size**2
		dphidt = -self.M1*self.ksq*mu + birth_death
		return dphidt

	def _noisy_delta(self):
		dW = mkl_fft.fft2(np.random.normal(size=(self.size, self.size)).astype('complex128'))
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
		theta = np.arctan((y-midpoint)/(x-midpoint))
		radius = radius + skew*np.cos(theta*2)
		phi = 0.7*(- np.tanh((np.sqrt((x-midpoint)**2+(y-midpoint)**2)-radius)/l)+1)
		phi[midpoint, midpoint] = 0.7*2
		phi += self.phi_target
		return mkl_fft.fft2(phi)

	def plot_slice(self, label, n=-1):
		phi = self.phi[n]
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		low, high = -1, 1
		plt.imshow(phi, vmin=low, vmax=high, cmap='seismic')
		plt.colorbar()
		plt.tight_layout()
		plt.savefig(label+"_snapshot_{}.pdf".format(n))
		plt.close()


	def make_movie(self, label, t_grid=1):
		fig = plt.figure()
		low, high = -1, 1
		ims = []
		im = plt.imshow(self.phi[0], vmin=low, vmax=high, animated=True, cmap='seismic')
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

	def make_bd_movie(self, label):
		fig = plt.figure()
		bd = - (self.phi+self.phi_shift)*(self.phi-self.phi_target)
		high = max(-np.min(bd), np.max(bd))
		low = -high
		ims = []
		im = plt.imshow(bd[0], vmin=low, vmax=high, animated=True, cmap='seismic')
		plt.colorbar(im)
		for i in range(self.n_batches):
			xy = bd[i]
			im = plt.imshow(xy, vmin=low, vmax=high, animated=True, cmap='seismic')
			ims.append([im])
		ani = am.ArtistAnimation(fig, ims, interval=100, blit=True,
										repeat_delay=1000)
		mywriter = am.FFMpegWriter()
		ani.save(label+"_bd_movie.mp4", writer=mywriter)
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
