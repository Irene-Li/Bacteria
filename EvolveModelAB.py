import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.fftpack import fft, ifft, fftfreq
import time
from DetEvolution1D import DetEvolution1D
from Entropy import *

class EvolveModelAB(DetEvolution1D):
    def __init__(self, M1=None, k=None, M2=None, delta_b=None):
        self.k = k
        self.M1 = M1
        self.a = M1 # so that random init in super class works
        self.M2 = M2
        self.delta_b = delta_b

    def load(self, label):
        self.phi = np.load('{}_data.npy'.format(label))

        with open('{}_params.json'.format(label), 'r') as f:
            params = json.load(f)
        self.k = params['k']
        self.M1 = params['M1']
        self.M2 = params['M2']
        self.delta_b = params['delta_b']
        self.X = params['X']
        self.T = params['T']
        self.dt = params['dt']
        self.dx = params['dx']
        self.size = params['size']
        self.n_batches = params['n_batches']
        self.step_size = params['step_size']

    def save(self, label):
        np.save("{}_data.npy".format(label), self.phi)

        params = {
            'T': self.T,
            'dt': self.dt,
            'dx': self.dx,
            'X': self.X,
            'n_batches': self.n_batches,
            'step_size': self.step_size,
            'size': self.size,
            'k': self.k,
            'M1': self.M1,
            'M2': self.M2,
            'delta_b': self.delta_b
        }

        with open('{}_params.json'.format(label), 'w') as f:
            json.dump(params, f)

    def _modify_params(self):
        # Want dx=1
        length_ratio = 1/self.dx
        self.dx = 1
        self.X = self.X * length_ratio
        self.M1 *= length_ratio**2
        self.k *= length_ratio**2

        # Want M1*k= 1
        time_ratio = self.M1 * self.k
        self.M1 /= time_ratio
        self.T = self.T * time_ratio
        self.M2 /= time_ratio
        self.dt = self.dt * time_ratio
        self.step_size = self.step_size * time_ratio

        print(self.k)
        print(self.M1)
        print(self.M2)

    def rescale_to_standard(self):
        # Want M1 = k = 1
        space_ratio = np.sqrt(1/self.k)
        time_ratio = self.k/self.M1
        self.dx = space_ratio * self.dx
        self.X = space_ratio * self.X
        self.T = time_ratio * self.T
        self.dt = time_ratio * self.dt
        self.step_size = time_ratio * self.step_size
        self.M1 = 1
        self.k = 1
        self.M2 = self.M2/time_ratio

    def evolve(verbose=False):
        super().evolve(verbose=verbose, ps=False)

    def _fd_delta_phi(self, t, phi):
        mu1 = - phi + phi**3 - self.k * self._laplacian(phi)
        lap_mu1 = self._laplacian(mu1)
        mu2 = (1+self.delta_b)*phi**3 + phi
        delta = self.M1*lap_mu1 - self.M2*mu2

        return delta

    def _fd_jac(self, t, phi):
        temp = sp.diags([(3*phi**2 - 1)], [0], shape=(self.size, self.size))
        jac = self.M1*(self._laplacian_sparse * temp - self.k * self._laplacian_sparse * self._laplacian_sparse)
        temp_diag = self.M2*(1 + 3*(1+self.delta_b)*phi**2)
        jac -= sp.diags([temp_diag], [0], shape=(self.size, self.size))
        return jac.todense()


class EntropyModelAB(EntropyProductionFourier):

    def __init__(self):
        pass

    def load(self, label):
        phi = np.load('{}_data.npy'.format(label))
        self.final_phi = phi[-2]

        with open('{}_params.json'.format(label), 'r') as f:
            params = json.load(f)
        self.k = params['k']
        self.M1 = params['M1']
        self.M2 = params['M2']
        self.delta_b = params['delta_b']
        self.X = params['X']
        self.T = params['T']
        self.dt = params['dt']
        self.dx = params['dx']
        self.size = params['size']
        self.n_batches = params['n_batches']
        self.step_size = params['step_size']


    def entropy_with_modelAB(self, current=False, reg=1):
        self._make_laplacian_matrix()
        if current:
            self._entropy_with_modelAB_currents()
        else:
            self.calculate_entropy(reg=reg)


    def _entropy_with_modelAB_currents(self):
        self._make_gradient_matrix()
        final_phi_fourier = fft(self.final_phi)
        final_phi_cube_fourier = fft(self.final_phi**3)
        self._mu1_fourier = (-final_phi_fourier + final_phi_cube_fourier)
        self._mu1_fourier -= self.k*self._laplacian_fourier*final_phi_fourier
        self._mu2_spatial = (self.final_phi + (1+self.delta_b)*self.final_phi**3)
        J_1 = self.M1 * self._gradient_fourier * self._mu1_fourier
        J_1 = ifft(J_1)
        J_2 = self.M2 * self._mu2_spatial

        entropy_from_model_B = J_1*J_1/self.M1
        entropy_from_model_A= J_2*J_2/self.M2
        self.entropy = entropy_from_model_A + entropy_from_model_B
        print(np.sum(self.entropy))

    def _make_first_order_matrix(self):
        A = self._fft_matrix(np.diag(self.final_phi**2))
        diag = 3*self.M1*self._laplacian_fourier - 3*self.M2*(1+self.delta_b)
        self.first_order_matrix_orig = np.einsum('i, ij -> ij', diag, A)
        self.first_order_matrix_orig = sp.csr_matrix(self.first_order_matrix_orig)
        diag = - self.M1*self._laplacian_fourier - self.M1*self.k*self._laplacian_fourier**2 - self.M2
        self.first_order_matrix_orig += sp.diags([diag], [0], shape=(self.size, self.size))
        self.first_order_matrix = self.first_order_matrix_orig.todense()

    def _make_noise_matrix(self):
        diag = -2*self.M1*self._laplacian_fourier + 2*self.M2
        self.noise_matrix = sp.diags([diag], [0], shape=(self.size, self.size))

if __name__ == "__main__":

    # Parameters
    M1 = 0.2
    k = 5
    X = 100
    dx = 1
    dt = 1e-3
    n_batches = 100
    M2 = 5e-6
    T = 10/M2
    delta_b = 0.1
    flat = True

    # Label for the run
    label = 'M2_{}_delta_b_{}'.format(M2, delta_b)
    print(label)
    
    start_time = time.time()
    solver = EvolveModelAB(M1, k, M2, delta_b)
    solver.initialise(X, dx, T, dt, n_batches, 0, flat=flat)
    solver.evolve()
    solver.save(label)
    end_time = time.time()
    print('The simulation took: ')
    print(end_time - start_time)

    solver.plot_steady_state(label)

    # solver = EntropyModelAB()
    # solver.load(label)
    #
    # # solver.read_entropy(label)
    # opt = 'currents'
    #
    # solver.entropy_with_modelAB(option=opt)
        # solver.plot_entropy(label+'_'+opt)
