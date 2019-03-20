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
            assert np.abs(phi_target - 0) < 1e-6
        super().__init__(a, k, u, phi_target, phi_shift)

    def _fd_delta_phi(self, t, phi):
        mu1 = self.a * ( - phi + phi ** 3) - self.k * self._laplacian(phi)
        lap_mu1 = self._laplacian(mu1)
        J2 = self.u * self.phi_shift * (phi + phi**3)
        delta = lap_mu1 - J2

        return delta

    def _fd_jac(self, t, phi):
        temp = sp.diags([self.a *(3 * phi**2 - 1)], [0], shape=(self.size, self.size))
        jac = self._laplacian_sparse * temp - self.k * self._laplacian_sparse * self._laplacian_sparse
        temp_diag = self.u * self.phi_shift * (1 + 3 * phi**2)
        jac -= sp.diags([temp_diag], [0], shape=(self.size, self.size))
        return jac.todense()


class EntropyModelAB(EntropyProductionFourier):

    def entropy_with_modelAB_currents(self):
        self._make_laplacian_matrix()
        self._make_gradient_matrix()
        final_phi_fourier = fft(self.final_phi)
        final_phi_cube_fourier = fft(self.final_phi**3)
        mu1 = self.a*(-final_phi_fourier + final_phi_cube_fourier) - self.k*self._laplacian_fourier*final_phi_fourier
        J_1 = self._gradient_fourier * mu1
        J_1 = ifft(J_1)

        J_2 = self.u * self.phi_shift * (self.final_phi + self.final_phi**3)
        M_2 = self.u * self.phi_shift /2

        self.entropy_from_model_B_current = J_1*J_1
        self.entropy_from_model_A_current = J_2*J_2/M_2
        self.entropy = self.entropy_from_model_A_current + self.entropy_from_model_B_current

if __name__ == "__main__":

    # Label for the run
    label = 'X_50_u_1e-6'

    # # Parameters
    # a = 0.2
    # k = 1
    # X = 50
    # dx = 0.1
    # dt = 1e-3
    # n_batches = 100
    # u = 1e-6
    # T = 1/u
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

    solver = EntropyProductionFourier()
    solver.load(label)

    # solver.read_entropy(label)

    # solver.compare_entropy()
    # solver.entropy_with_modelAB_currents()
    # solver.plot_entropy_from_modelAB_currents(label)

    solver.entropy_with_modelAB_currents()
    solver.plot_entropy_from_modelAB_currents(label) 
