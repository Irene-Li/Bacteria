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
		time_ratio = length_ratio**4 * (self.k * self.M1)
        self.M1 /= time_ratio
		self.T = self.T * time_ratio
		self.M2 /= time_ratio
		self.dt = self.dt * time_ratio
		self.step_size = self.step_size * time_ratio

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
			'u': self.u,
			'phi_shift': self.phi_shift,
			'phi_target': 0
		}
		return params

	def rescale_to_standard(self):
        # Want M1*a = M1*k = 1
		time_ratio = (self.M1*self.a)**2/self.k
		space_ratio = np.sqrt(self.a/self.k)
		self.dx = space_ratio * self.dx
		self.X = space_ratio * self.X
		self.T = time_ratio * self.T
		self.dt = time_ratio * self.dt
		self.step_size = time_ratio * self.step_size
		self.M1 = self.M1 * space_ratio**2 / time_ratio
        self.a
		self.u = self.u/time_ratio

    def _fd_delta_phi(self, t, phi):
        mu1 = self.a * ( - phi + phi ** 3) - self.k * self._laplacian(phi)
        lap_mu1 = self._laplacian(mu1)
        J2 = self.u * self.phi_shift/2 * self.a * (phi + phi ** 3)
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

        J_2 = self.u * self.phi_shift/2 * self.a *(self.final_phi + self.final_phi**3)
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
