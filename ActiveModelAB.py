import numpy as np
import time
import json
from Bacteria.StoEvolution2D import *
from . import pseudospectral as ps 
import pygl 

rng = np.random.default_rng() 


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
        pass 
        # length_ratio = self.dx
        # time_ratio = length_ratio**4/self.k #such that M1*k=1

        # self.dx = 1
        # self.X /= length_ratio
        # self.T /= time_ratio
        # self.dt /= time_ratio
        # self.step_size /= time_ratio

        # self.k /= length_ratio**2
        # self.lbda /= length_ratio**2
        # self.zeta /= length_ratio**2
        # self.M1 *= time_ratio/length_ratio**2
        # self.u *= time_ratio
        # self.M2 *= time_ratio
        # self.epsilon /= length_ratio*length_ratio*time_ratio

    def print_params(self):
        super().print_params()
        print('lambda', self.lbda, '\n',
        'zeta', self.zeta, '\n')

    def evolve(self, verbose=True, cython=False, fd=True):
        if cython:
            nitr = int(self.T/self.dt)
            if fd: 
                raise Exception('Not implemented')
            else: 
                self.phi = ps.evolve_sto_ps_active(np.fft.fft2(self.phi_initial), self.M1, self.a, self.k, self.u,
                                        self.phi_shift, self.phi_target, self.lbda, self.zeta,
                                        self.epsilon, self.dt, nitr, self.n_batches, self.X)
        else:
            if fd: 
                print('Using finite difference ')
                self.naive_evolve_fd(verbose)
            else: 
                self.naive_evolve_ps(verbose)

    def naive_evolve_ps(self, verbose):
        self._make_k_grid()
        self._make_filters()
        self.phi = np.zeros((self.n_batches, self.size, self.size))
        phi = np.fft.fft2(self.phi_initial)

        n = 0
        for i in range(int(self.T/self.dt)):
            if i % self.batch_size == 0:
                phi_x = np.real(mkl_fft.ifft2(phi))
                self.phi[n] = phi_x
                if verbose:
                    print('iteration: {}    mean: {}'.format(n, phi[0, 0].real/(self.X*self.X)))
                n += 1
            delta = self._delta_ps(phi)*self.dt 
            noisy_delta = self._noisy_delta()
            phi += delta + noisy_delta


    def naive_evolve_fd(self, verbose): 
        grid = {'dim':2, 'Nx':self.size, 'Ny':self.size}
        self.fd = pygl.utils.FiniteDifference(grid)
        self.noise_amp = np.sqrt(2*self.epsilon*self.M1/self.dt)
        self.phi = np.zeros((self.n_batches, self.size, self.size))
        phi = self.phi_initial 

        n = 0 
        for i in range(int(self.T/self.dt)): 
            if i % self.batch_size == 0: 
                self.phi[n] = phi 
                if verbose: 
                    print('iteration: {}  mean: {}'.format(n, np.sum(phi)/(self.X*self.X)))
                n += 1
            phi += self._delta_fd(phi)*self.dt 


    def _delta_fd(self, phi): 
        dphidx = self.fd.diffx(phi)
        dphidy = self.fd.diffy(phi)
        lap_phi = self.fd.laplacian(phi)
        lambda_term = self.lbda*(dphidx*dphidx+dphidy*dphidy)
        mu = self.a*(-phi+phi*phi*phi) - self.k*lap_phi 
        Nx = self.noise_amp*rng.standard_normal((self.size, self.size))
        Ny = self.noise_amp*rng.standard_normal((self.size, self.size))
        Jx = - self.M1*self.fd.diffx1(mu) + Nx
        Jy = - self.M1*self.fd.diffy1(mu) + Ny 
        Jx_neq = self.M1*(self.zeta*lap_phi*dphidx - self.fd.diffx1(lambda_term))
        Jy_neq = self.M1*(self.zeta*lap_phi*dphidy - self.fd.diffy1(lambda_term))
        dphidt1 = - self.fd.diffx1(Jx) - self.fd.diffy1(Jy) - self.fd.diffx(Jx_neq) - self.fd.diffy(Jy_neq)

        N = np.sqrt(2*self.M2*self.epsilon/self.dt)*rng.standard_normal((self.size, self.size))
        birth_death = - self.u*(phi+self.phi_shift)*(phi-self.phi_target) + N 
        return birth_death+dphidt1 

    def _delta_ps(self, phi):
        phi_x = np.fft.ifft2(phi)
        self.phi_sq = np.fft.fft2(phi_x*phi_x)
        self.phi_cube = np.fft.fft2(phi_x*phi_x*phi_x)
        np.putmask(self.phi_cube, self.dealiasing_triple, 0)
        np.putmask(self.phi_sq, self.dealiasing_double, 0)

        # lambda term 
        dphidx = np.fft.ifft2(1j*self.kx*phi)
        dphidy = np.fft.ifft2(1j*self.ky*phi)
        lambda_term = np.fft.fft2(dphidx*dphidx + dphidy*dphidy)
        np.putmask(lambda_term, self.dealiasing_double, 0)

        # zeta term 
        lap_phi = np.fft.ifft2(-self.ksq*phi)
        Jx = np.fft.fft2(self.zeta*lap_phi*dphidx) 
        Jy = np.fft.fft2(self.zeta*lap_phi*dphidy)
        zeta_term = -1j*self.kx*Jx -1j*self.ky*Jy
        np.putmask(zeta_term, self.dealiasing_double, 0)


        mu = self.a*(-phi+self.phi_cube) + self.k*self.ksq*phi + self.lbda*lambda_term 
        birth_death = - self.u*(self.phi_sq+(self.phi_shift-self.phi_target)*phi)
        birth_death[0, 0] += self.u*self.phi_shift*self.phi_target*self.size**2
        dphidt = self.M1*(-self.ksq*mu + zeta_term) + birth_death
        return dphidt








