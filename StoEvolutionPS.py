import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as am
import time
import scipy.sparse as sp
from scipy.fftpack import fft2, ifft2, fftfreq
import json
from StoEvolution import *



class StoEvolutionPS(StoEvolution):

    def evolve(self, verbose=True):
        self._make_k_grid()
        self._make_filters()
        self.phi = np.zeros((self.n_batches, self.size, self.size))
        phi = self.phi_initial

        n = 0
        for i in range(int(self.T/self.dt)):
            if i % self.batch_size == 0:
                self.phi[n] = np.real(ifft2(phi))
                if verbose:
                    print('iteration: {}	mean: {}'.format(i, phi[0, 0]/(self.size**2)))
                n += 1
            phi += self._delta(phi)*self.dt +self._noisy_delta()

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
        phi[0, 1] = self.size*self.size*1 # blows up
        return phi

    def make_movie(self, label, t_grid=1):
        fig = plt.figure()
        low, high = -1.2, 1.2
        ims = []
        im = plt.imshow(self.phi[0], vmin=low, vmax=high, animated=True)
        plt.colorbar(im)
        for i in range(self.n_batches):
            xy = self.phi[i]
            im = plt.imshow(xy, vmin=low, vmax=high, animated=True)
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
    phi_t = 0
    phi_shift = 100

    X = 128
    dx = 1
    T = 1e4
    dt = 5e-3
    n_batches = 100
    initial_value = 0
    flat = True

    for phi_t in [-0.4, -0.5, -0.6, -0.7]:
    	label = 'u_{}_phi_t_{}'.format(u, phi_t)

    	start_time = time.time()
    	solver = StoEvolutionPS(epsilon, a, k, u, phi_t, phi_shift)
    	solver.initialise(X, dx, T, dt, n_batches, initial_value, flat=flat)
    	solver.save_params(label)
    	solver.print_params()
    	solver.evolve(verbose=True)
    	solver.save_phi(label)
    	end_time = time.time()
    	print('The simulation took: {}'.format(end_time - start_time))
