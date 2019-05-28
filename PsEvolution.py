import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import ode


class PsEvolution(TimeEvolution):

	def evolve(self, verbose=True):
        self._make_k_grid()
        self._make_filters()
        self.phi = np.zeros((self.n_batches, self.size, self.size))
        phi = self.phi_initial

        small_batch = self.batch_size
		while small_batch > 1000:
			small_batch /= 10 # decrease the amount of time integration at each step

		r = ode(self._delta).set_integrator('lsoda', atol=1e-10, nsteps=small_batch)
		r.set_initial_value(self.phi_initial, 0)

		n = 0
		phi = self.phi_initial

		for i in range(int((self.T/self.dt)/small_batch)):
			if r.successful():
				if i % int(self.batch_size/small_batch) == 0:
					self.phi[n] = phi
					print('iteration: {}	mean: {}'.format(n*self.batch_size, np.mean(self.phi[n])))
					n += 1
				phi = r.integrate(r.t+self.dt*small_batch)

    def _make_k_grid(self):
        Nx, Ny = self.size, self.size
        kx = fftfreq(Nx)*2*np.pi
        ky = fftfreq(Ny)*2*np.pi
        self.kx, self.ky = np.meshgrid(kx, ky)
        self.ksq = self.kx*self.kx + self.ky*self.ky

    def _make_filters(self):
        kk1 = self.kx
        kmax = np.max(np.abs(kk1))
        filtr = np.ones_like(kk1)
        filtr2 = np.ones_like(kk1)
        filtr[np.where(np.abs(kk1)>kmax*2./3)] = 0.
        filtr2[np.where(np.abs(kk1)>kmax*1./2)] = 0.

        kk1 = self.ky
        kmax = np.max(np.abs(kk1))
        filtr_1 = np.ones_like(kk1)
        filtr_12 = np.ones_like(kk1)
        filtr_1[np.where(np.abs(kk1)>kmax*2./3)] = 0.
        filtr_12[np.where(np.abs(kk1)>kmax*1./2)] = 0.

        self.dealiasing_double = filtr*filtr_1
        self.dealiasing_triple = filtr2*filtr_12

    def _delta(self, phi):
        phi_x = ifft2(phi)
        phi_cube = self.dealiasing_triple*fft2(phi_x**3)
        phi_sq = self.dealiasing_double*fft2(phi_x**2)

        mu = self.a*(-phi+phi_cube) + self.k*self.ksq*phi
        birth_death = - self.u*(phi_sq+(self.phi_shift-self.phi_target)*phi)
        birth_death[0] += self.u*self.phi_shift*self.phi_target*self.size**2
        dphidt = -self.M1*self.ksq*mu + birth_death
        return dphidt

    def _noisy_delta(self):
        dWx = fft2(np.random.normal(0, 1, (self.size, self.size)))
        dWy = fft2(np.random.normal(0, 1, (self.size, self.size)))
        dW = fft2(np.random.normal(0, 1, (self.size, self.size)))

        noise = 1j*np.sqrt(2*self.M1*self.epsilon*self.dt)*(self.kx*dWx + self.ky*dWy)
        noise += np.sqrt(2*self.M2*self.epsilon*self.dt)*dW
        return noise

    def _flat_surface(self, initial_value):
        return np.zeros((self.size, self.size)) + initial_value + 0j

    def _sin_surface(self, initial_value):
        phi = np.zeros((self.size, self.size)) + 0j
        phi[0, 1] = self.size*self.size*0.1
        return phi

    def make_movie(self, label, t_grid=1):
        fig = plt.figure()
        ims = []
        for i in range(self.n_batches):
            xy = self.phi[i]
            im = plt.imshow(xy, animated=True)
            ims.append([im])
        ani = am.ArtistAnimation(fig, ims, interval=100, blit=True,
                                        repeat_delay=1000)
        mywriter = am.FFMpegWriter()
        ani.save(label+"_movie.mp4", writer=mywriter)
