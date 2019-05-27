import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy.sparse as sp
import scipy.linalg as sl
import scipy.sparse.linalg
from scipy.fftpack import fft, ifft, fftfreq
import json
from TimeEvolution import TimeEvolution

class EntropyProduction(TimeEvolution):

	# ============
	# IO functions
	# ============

	def load(self, label):
		super(EntropyProduction, self).load(label)
		self.final_phi = self.phi[-2]

	def write_entropy(self, label):
		np.save("{}_entropy.npy".format(label), self.entropy)

	def read_entropy(self, label):
		filename = "{}_entropy.npy".format(label)
		if os.path.isfile(filename):
			self.entropy = np.load(filename)
		else:
			self.calculate_entropy()
			self.write_entropy(label)


	# =============
	# Main function
	# =============

	def calculate_entropy():
		# to be implemented in subclasses
		pass

	# ==================
	# Plotting functions
	# ==================

	def plot_entropy(self, label):
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif', size=12)

		x = np.arange(0, (self.size)* self.dx, self.dx)

		plt.subplot(2, 1, 1)
		plt.plot(x, np.real(self.entropy), 'k-')
		plt.ylabel(r"$\dot{S}$")
		plt.subplot(2, 1, 2)
		plt.plot(x, self.final_phi, 'k-')
		plt.ylabel(r"$\phi$")
		plt.xlabel(r"$x$")
		plt.tight_layout()
		plt.savefig("{}_entropy.pdf".format(label))
		plt.close()

	# ================
	# Helper functions
	# ================

	def _make_correlation_matrix(self):
		self.correlation_matrix = sl.solve_lyapunov(self.first_order_matrix, (-self.noise_matrix).todense())

		# Check accuracy of the Lyapunov eq
		temp = self.first_order_matrix.dot(self.correlation_matrix) + self.correlation_matrix.dot(self.first_order_matrix.T.conj())
		print("Error in Lyapunov eq: ", sl.norm(temp + self.noise_matrix.todense())/sp.linalg.norm(self.noise_matrix))
		print("Norm of A: ", sl.norm(self.first_order_matrix))
		print("Norm of C: ", sl.norm(self.correlation_matrix))
		print("Norm of K: ", sp.linalg.norm(self.noise_matrix))

		C_subspace = self._project_matrix(self.correlation_matrix)
		A_subspace = self._project_matrix(self.first_order_matrix_orig.todense())
		K_subspace = self._project_matrix(self.noise_matrix.todense())
		temp = A_subspace.dot(C_subspace) + C_subspace.dot(A_subspace.T.conj())
		print("Error in Lyapunov eq in subspace: ", sl.norm(temp + K_subspace)/sl.norm(K_subspace))

	def _compare_translational_mode(self):
		eigenvalues, eigenvectors = sl.eig(self.first_order_matrix)
		sorted_indices = np.argsort(eigenvalues)
		eigenvectors = eigenvectors[:, sorted_indices[-3:]]
		eigenvalues = eigenvalues[sorted_indices[-3:]]
		plt.plot(self.final_phi, label="phi")
		for (i, eig) in enumerate(eigenvalues):
			label = "{0:2f}".format(np.real(eig))
			plt.plot(eigenvectors[:, i], label=label)
		plt.legend()
		plt.show()


	def _add_to_translational_dof(self, reg=1):
		eigenvalues, eigenvectors = sl.eig(self.first_order_matrix)
		max_index = np.argmax(eigenvalues)
		goldstone_mode = eigenvectors[:, max_index]
		self.projection_onto_gm = np.outer(goldstone_mode, goldstone_mode.conj())
		self.first_order_matrix -= self.projection_onto_gm * reg
		self.projection_out_gm = np.identity(self.size) - self.projection_onto_gm

	def _project_matrix(self, matrix):
		return self.projection_out_gm.dot(matrix.dot(self.projection_out_gm))


class EntropyProductionFourier(EntropyProduction):

	def __init__(self, quad_bd=True):
		if quad_bd:
			self._make_first_order_matrix = self._make_first_order_matrix_quad_bd
			self._make_noise_matrix = self._make_noise_matrix_quad_bd
		else:
			self._make_first_order_matrix = self._make_first_order_matrix_lin_bd
			self._make_noise_matrix = self._make_noise_matrix_lin_bd

	def calculate_entropy(self):
		reg = 1

		self._make_laplacian_matrix()
		self._make_first_order_matrix()
		self._make_noise_matrix()
		self._add_to_translational_dof(reg=reg)
		self._make_correlation_matrix()


		S = self._calculate_entropy_with_ss_dist()
		S_real = self._ifft_matrix(S)
		self.entropy = S_real.diagonal()
		print("total entropy production: ", np.sum(self.entropy))

	def compare_entropy(self):
		reg = 1

		self._make_laplacian_matrix()
		self._make_first_order_matrix()
		self._make_noise_matrix()
		self._add_to_translational_dof(reg=reg)
		self._make_correlation_matrix()

		S1 = self._calculate_entropy_with_conjugate_currents()
		S2 = self._calculate_entropy_with_antisym_A()
		print("total entropy production of S1:", np.trace(S1))
		print("total entropy production of S2:", np.trace(S2))
		S1_real = self._ifft_matrix(S1)
		S2_real = self._ifft_matrix(S2)
		plt.plot(np.diag(S1_real), label='S1')
		plt.plot(np.diag(S2_real), label='S2')
		plt.plot(np.imag(np.diag(S2_real)), label='S2 imag')
		plt.legend()
		plt.show()

	def entropy_with_modelAB(self):
		reg = 1

		self._make_laplacian_matrix()
		self._make_first_order_matrix()
		self._make_gradient_matrix()
		self._make_noise_matrix()
		self._add_to_translational_dof(reg=reg)
		self._make_correlation_matrix()

		final_phi_fourier = fft(self.final_phi)
		final_phi_cube_fourier = fft(self.final_phi**3)
		mu1 = self.a*(-final_phi_fourier + final_phi_cube_fourier) - self.k*self._laplacian_fourier*final_phi_fourier
		mu2 = self.u*(self.final_phi + self.phi_shift)*(self.final_phi - self.phi_target)

		C = self._project_matrix(self.correlation_matrix)
		C_reg = C + self.projection_onto_gm
		C_inv = sl.inv(C_reg)
		C_inv = self._project_matrix(C_inv)
		C_inv_phi = C_inv.dot(final_phi_fourier)

		J_1 = self._gradient_fourier * (mu1 - C_inv_phi)
		J_1 = ifft(J_1)

		J_2 = mu2 - ifft(C_inv_phi)
		M_2 = self.u*(self.phi_shift+self.phi_target/2)

		self.entropy_from_model_B_current = J_1*J_1
		self.entropy_from_model_A_current = J_2*J_2/M_2
		self.entropy = self.entropy_from_model_A_current + self.entropy_from_model_B_current

	def _calculate_entropy_with_conjugate_currents(self):
		# Take the square root of the noise matrix
		K_diag = self.noise_matrix.diagonal()/2
		sqrt_K = np.sqrt(K_diag)

		C = self._project_matrix(self.correlation_matrix)
		C_reg = C + self.projection_onto_gm
		C_inv = sl.inv(C_reg)
		C_inv = self._project_matrix(C_inv)

		B = - np.einsum('i, ij->ij', 1/K_diag, self.first_order_matrix_orig.todense())
		E = np.einsum('i, ij->ij', sqrt_K, B - C_inv)
		S = E.dot(C.dot(E.T.conj()))

		return S

	def _calculate_entropy_with_ss_dist(self):
		K_diag = self.noise_matrix.diagonal()/2
		C = self._project_matrix(self.correlation_matrix)
		C_reg = C + self.projection_onto_gm
		C_inv = sl.inv(C_reg)
		C_inv = self._project_matrix(C_inv)

		A = - self.first_order_matrix_orig.todense()
		B = np.einsum('i, ij->ij', 1/K_diag, A)
		E = B - C_inv
		K_dot_E = np.einsum('i,ij->ij', K_diag, E)

		S = K_dot_E.dot(C.dot(E.T.conj()))
		return S

	def _make_first_order_matrix_lin_bd(self):
		A = self._fft_matrix(np.diag(self.final_phi**2))
		self.first_order_matrix_orig = 3 * self.a * np.einsum('i, ij -> ij', self._laplacian_fourier, A)
		self.first_order_matrix_orig = sp.csr_matrix(self.first_order_matrix_orig)
		diag = - self.a * self._laplacian_fourier - self.k * self._laplacian_fourier ** 2 - self.u
		self.first_order_matrix_orig += sp.diags([diag], [0], shape=(self.size, self.size))
		self.first_order_matrix = self.first_order_matrix_orig.todense()

	def _make_first_order_matrix_quad_bd(self):
		A = self._fft_matrix(np.diag(self.final_phi**2))
		self.first_order_matrix_orig = 3 * self.a * np.einsum('i, ij -> ij', self._laplacian_fourier, A)
		self.first_order_matrix_orig = sp.csr_matrix(self.first_order_matrix_orig)
		diag = - self.a * self._laplacian_fourier - self.k * self._laplacian_fourier ** 2 - self.u * (self.phi_shift - self.phi_target)
		self.first_order_matrix_orig += sp.diags([diag], [0], shape=(self.size, self.size))
		A = self._fft_matrix(np.diag(2 * self.u * self.final_phi))
		self.first_order_matrix_orig -= sp.csr_matrix(A)
		self.first_order_matrix = self.first_order_matrix_orig.todense()

	def _make_laplacian_matrix(self):
		x = np.arange(self.size)
		self._laplacian_fourier = - 2 * (1 - np.cos(2 * np.pi * x/self.size))

	def _make_gradient_matrix(self):
		self._gradient_fourier = np.sqrt(- self._laplacian_fourier)*(-1j)
		n = int(self.size/2)+1
		self._gradient_fourier[n:] *= (-1)

	def _fft_matrix(self, matrix):
		matrix_fft = fft(matrix).T.conj()
		matrix_fft = fft(matrix_fft).T.conj()/self.size
		return matrix_fft

	def _ifft_matrix(self, matrix):
		matrix_ifft = ifft(matrix).T.conj()
		matrix_ifft = ifft(matrix_ifft).T.conj() * self.size
		return matrix_ifft

	def _make_noise_matrix_quad_bd(self):
		diag = -2*self._laplacian_fourier + self.u*(2*self.phi_shift+self.phi_target)
		self.noise_matrix = sp.diags([diag], [0], shape=(self.size, self.size))

	def _make_noise_matrix_lin_bd(self):
		diag = -2*self._laplacian_fourier + self.u
		self.noise_matrix = sp.diags([diag], [0], shape=(self.size, self.size))










if __name__ == "__main__":

	label = 'X_100_u_1e-6_random_init'

	solver = EntropyProductionFourier()
	solver.load(label)

	# solver.read_entropy(label)

	# solver.compare_entropy()
	# solver.entropy_with_modelAB()
	# solver.plot_entropy_from_modelAB_currents(label)

	# solver.calculate_entropy()
	solver.entropy_with_modelAB()
	# solver.read_entropy(label)
	solver.plot_entropy(label+'new')
	solver.write_entropy(label+'new')
