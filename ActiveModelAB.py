import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as am
import time
import scipy.sparse as sp
from scipy.fftpack import fftfreq
import json
from StoEvolution1D import *
from pseudospectral import evolve_sto_ps_active
import mkl_fft


class ActiveModelAB(StoEvolution2D):

    def __init__(self, epsilon = None, a=None, k=None, u=None, phi_target=None, phi_shift=None, lbda=None, zeta=None):
        super().__init__(epsilon, a, k, u, phi_target, phi_shift)
        self.lbda = lbda
        self.zeta = zeta

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
            'lambda': self.lbda,
            'zeta': self.zeta
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
        self.lbda = params['lambda']
        self.zeta = params['zeta']
        self.X = params['X']
        self.T = params['T']
        self.dt = params['dt']
        self.dx = params['dx']
        self.size = params['size']
        self.n_batches = params['n_batches']
        self.step_size = params['step_size']
        self.M1 = 1/self.k
        self.M2 = self.u*(self.phi_shift+self.phi_target/2)


    def _modify_params(self):
        length_ratio = self.dx
        time_ratio = length_ratio**4/self.k #such that M1*k=1

        self.dx = 1
        self.X /= length_ratio
        self.T /= time_ratio
        self.dt /= time_ratio
        self.step_size /= time_ratio

        self.k /= length_ratio**2
        self.lbda /= length_ratio**2
        self.zeta /= length_ratio**2
        self.M1 *= time_ratio/length_ratio**2
        self.u *= time_ratio
        self.M2 *= time_ratio
        self.epsilon /= length_ratio

    def print_params(self):
        super().print_params()
        print('lambda', self.lbda, '\n',
		'zeta', self.zeta, '\n')

    def evolve(self, verbose=True, cython=True):
		self.phi_initial = mkl_fft.fft2(self.phi_initial)
		if cython:
			print('not implemented')
		else:
			self.naive_evolve(verbose)

    def _delta(self, phi):
		phi_x = mkl_fft.ifft2(phi)
		self.phi_sq = mkl_fft.fft2(phi_x**2)
		self.phi_cube = mkl_fft.fft2(phi_x**3)
		np.putmask(self.phi_cube, self.dealiasing_triple, 0)
		np.putmask(self.phi_sq, self.dealiasing_double, 0)


        dphidx = mkl_fft.ifft2(self.kx*phi)
        dphidy = mkl_fft.ifft2(self.ky*phi)



		mu = self.a*(-phi+self.phi_cube) + self.k*self.ksq*phi
		birth_death = - self.u*(self.phi_sq+(self.phi_shift-self.phi_target)*phi)
		birth_death[0, 0] += self.u*self.phi_shift*self.phi_target*self.size**2
		dphidt = -self.M1*self.ksq*mu + birth_death
		return dphidt
