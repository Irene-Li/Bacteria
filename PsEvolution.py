import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as am
import time
from scipy.integrate import ode
from scipy.fftpack import fft2, ifft2, fftfreq
from TimeEvolution import *


class PsEvolution(TimeEvolution):

	def evolve(self, verbose=True):
		self._make_k_grid()
		self._make_filters()
		self.phi = np.zeros((self.n_batches, self.size, self.size))
		phi = self.phi_initial

		small_batch = self.batch_size
		while small_batch > 10000:
			small_batch /= 10 # decrease the amount of time integration at each step

		r = ode(self._delta).set_integrator('vode', atol=1e-7, nsteps=small_batch)
		r.set_initial_value(self.phi_initial, 0)

		n = 0
		phi = self.phi_initial
		for i in range(int((self.T/self.dt)/small_batch)):
			if r.successful():
				if i % int(self.batch_size/small_batch) == 0:
					phi_complex = self._make_complex(phi)
					self.phi[n] = np.real(ifft2(phi_complex))
					print('iteration: {}	mean: {}'.format(n, phi[0]/self.size**2))
					n += 1
				phi = r.integrate(r.t+self.dt*small_batch)

	def _plot_state(self, phi, n):
		plt.imshow(phi)
		plt.colorbar()
		plt.savefig('state_{}.pdf'.format(n))
		plt.close()


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

	def _delta(self, t, phi):
		phi_complex = self._make_complex(phi)
		phi_x = ifft2(phi_complex)
		phi_cube = fft2(phi_x**3)
		phi_sq = fft2(phi_x**2)
		np.putmask(phi_cube, self.dealiasing_triple, 0)
		np.putmask(phi_sq, self.dealiasing_double, 0)

		mu = (-self.a+self.k*self.ksq)*phi_complex + self.a*phi_cube
		birth_death = - self.u*(phi_sq+(self.phi_shift-self.phi_target)*phi_complex)
		birth_death[0, 0] += self.u*self.phi_shift*self.phi_target*self.size**2
		dphidt_complex = -self.ksq*mu + birth_death
		return self._make_real(dphidt_complex)

	def _random_init(self, initial_value):
		init = np.random.normal(initial_value, np.sqrt(self.dt), (self.size, self.size, 2))
		init[0, 0, 1] = 0
		half_point = int(self.size/2+1)
		init[half_point, half_point, 1] = 0
		return np.ravel(init)

	def _sin_surface(self, initial_value):
		x = np.arange(self.size)
		y = np.arange(self.size)
		x, y = np.meshgrid(x, y)
		midpoint = int(self.size/2)
		size = 30
		l = np.sqrt(self.k/self.a)
		phi = - np.tanh((np.sqrt(1.2*(x-midpoint)**2+0.7*(y-midpoint)**2)-size)/l)
		phi_complex = fft2(phi)
		return self._make_real(phi_complex)

	def make_movie(self, label, t_grid=1):
		fig = plt.figure()
		ims = []
		low, high = -1.2, 1.2
		im = plt.imshow(self.phi[0], vmin=low, vmax=high, animated=True)
		plt.colorbar(im)
		for i in range(self.n_batches):
			xy = self.phi[i]
			im = plt.imshow(xy, animated=True)
			ims.append([im])
		ani = am.ArtistAnimation(fig, ims, interval=100, blit=True,
										repeat_delay=1000)
		mywriter = am.FFMpegWriter()
		ani.save(label+"_movie.mp4", writer=mywriter)


if __name__ == '__main__':

	epsilon = 0.1
	a = 0.2
	k = 1
	u = 1e-5
	phi_t = -0.8
	phi_shift = 10

	X = 128
	dx = 1
	T = 1e3
	dt = 1e-4
	n_batches = 100
	initial_value = -0.8
	flat = False

	for u in [1e-4]:
		label = 'u_{}_droplet'.format(u)

		start_time = time.time()
		solver = PsEvolution(a, k, u, phi_t, phi_shift)
		solver.initialise(X, dx, T, dt, n_batches, initial_value, flat=flat)
		solver.evolve(verbose=True)
		solver.save(label)
		end_time = time.time()
		print('The simulation took: {}'.format(end_time - start_time))
