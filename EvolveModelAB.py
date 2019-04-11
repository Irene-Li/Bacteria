import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.fftpack import fft, ifft, fftfreq
import time
from FdEvolution import FdEvolution
from Entropy import *

class EvolveModelAB(FdEvolution):
    def __init__(self, a=None, k=None, u=None, phi_target=None, phi_shift=None):
        if phi_shift:
            assert np.abs(phi_target) < 1e-6
        super().__init__(a, k, u, phi_target, phi_shift)
        self.M1 = 1
        self.M2 = self.u * self.phi_shift/2

    def load(self, label):
        super().load(label)
        self.M1 = 1
        self.M2 = self.u * self.phi_shift/2

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
        print(self.a)
        print(self.u)

    def _make_params_dict(self):
        params = {
			'T': self.T,
			'dt': self.dt,
			'dx': self.dx,
			'X': self.X,
			'n_batches': self.n_batches,
			'step_size': self.step_size,
			'size': self.size,
			'a': self.a * self.M1,
			'k': self.k * self.M1,
			'u': self.M2 * 2 /self.phi_shift,
			'phi_shift': self.phi_shift,
			'phi_target': 0
		}
        return params

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
        self.M2 = self.u/time_ratio

    def _fd_delta_phi(self, t, phi):
        mu1 =  self.a * ( - phi + phi**3) - self.k * self._laplacian(phi)
        lap_mu1 = self._laplacian(mu1)
        mu2 = self.a * phi**3 + 2 * phi
        delta = self.M1 * lap_mu1 - self.M2 * mu2

        return delta

    def _fd_jac(self, t, phi):
        temp = sp.diags([self.a *(3 * phi**2 - 1)], [0], shape=(self.size, self.size))
        jac = self.M1 * (self._laplacian_sparse * temp - self.k * self._laplacian_sparse * self._laplacian_sparse)
        temp_diag = self.M2 * 2 + 3 * self.M2 * self.a * phi**2
        jac -= sp.diags([temp_diag], [0], shape=(self.size, self.size))
        return jac.todense()


class EntropyModelAB(EntropyProductionFourier):

    def load(self, label):
        super().load(label)
        self.M1 = 1
        self.M2 = self.u * self.phi_shift/2

    def entropy_with_modelAB(self, option="currents"):
        self._make_laplacian_matrix()
        final_phi_fourier = fft(self.final_phi)
        final_phi_cube_fourier = fft(self.final_phi**3)
        self._mu1_fourier = self.a*(-final_phi_fourier + final_phi_cube_fourier)
        self._mu1_fourier -= self.k*self._laplacian_fourier*final_phi_fourier
        self._mu2_spatial = (2 * self.final_phi + self.a * self.final_phi**3)

        if option=="currents":
            self._entropy_with_modelAB_currents()
        elif option=="phi1_phi2":
            self._entropy_with_phi1_phi2()
        elif option=="psi":
            self._entropy_with_psi()
        else:
            print("Not a valid option")

    def _entropy_with_modelAB_currents(self):
        self._make_gradient_matrix()
        J_1 = self.M1 * self._gradient_fourier * self._mu1_fourier
        J_1 = ifft(J_1)
        J_2 = self.M2 * self._mu2_spatial

        entropy_from_model_B = J_1*J_1/self.M1
        entropy_from_model_A= J_2*J_2/self.M2
        self.entropy = entropy_from_model_A + entropy_from_model_B


    def _entropy_with_phi1_phi2(self):
        dtphi1 = self.M1 * ifft(self._laplacian_fourier * self._mu1_fourier)
        mu1_spatial = ifft(self._mu1_fourier)
        entropy_from_model_B = - dtphi1 * mu1_spatial
        entropy_from_model_A = self.M2 * self._mu2_spatial**2
        self.entropy = entropy_from_model_B + entropy_from_model_A


    def _entropy_with_psi(self):
        dtpsi_fourier = self.M1 * self._laplacian_fourier* self._mu1_fourier
        dtpsi_fourier += self.M2 * fft(self._mu2_spatial)
        noise_matrix = - self.M1*self._laplacian_fourier + self.M2
        dtpsi_spatial = ifft(dtpsi_fourier)
        other_part = dtpsi_fourier/noise_matrix
        other_part = ifft(other_part)
        self.entropy = other_part * dtpsi_spatial



if __name__ == "__main__":

    # Label for the run
    label = 'X_50_u_1e-6'

    # # Parameters
    # a = 0.5
    # k = 1
    # X = 50
    # dx = 0.1
    # dt = 1e-3
    # n_batches = 100
    # u = 8e-4
    # T = 10/u
    # phi_shift = 100
    # phi_target = 0
    # phi_init = 0
    # random = False
    #
    # start_time = time.time()
    # solver = EvolveModelAB(a, k, u, phi_target, phi_shift)
    # solver.initialise(X, dx, T, dt, n_batches, phi_init, random=random)
    # solver.evolve()
    # solver.save(label)
    # end_time = time.time()
    # print('The simulation took: ')
    # print(end_time - start_time)
    #
    # solver.rescale_to_standard()
    # solver.plot_steady_state(label, kink=0)

    solver = EntropyModelAB()
    solver.load(label)

    # solver.read_entropy(label)
    opt = 'currents'

    solver.entropy_with_modelAB(option=opt)
    solver.plot_entropy(label+'_'+opt)
