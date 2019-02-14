import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as sl
from scipy.fftpack import fft, ifft, fftfreq
import json
from TimeEvolution import TimeEvolution

class EntropyProduction(TimeEvolution):

	def calculate_entropy(self):
		reg = 1
		self.final_phi = self.phi[-2]

		self._make_laplacian_matrix()
		self._make_first_order_matrix()
		self._make_noise_matrix()
		self._add_to_translational_dof(reg=reg)
		self._make_correlation_matrix()

		self.correlation_matrix = sp.csr_matrix(self.correlation_matrix)
		self.entropy = self._multiply_in_fourier_space()


	def test(self):
		reg = 1
		self.final_phi = self.phi[-2]
		self.correlation_matrix = np.load("correlation.npy")
		self._make_laplacian_fourier()
		self._make_first_order_matrix_quad_bd()
		self._make_noise_matrix_quad_bd()
		self._add_to_translational_dof(reg=reg)


		identity = np.identity(self.size)
		one_minus_P = identity - self.projection

		C_subspace = one_minus_P.dot(self.correlation_matrix.dot(one_minus_P))
		A_subspace = one_minus_P.dot(self.first_order_matrix.dot(one_minus_P))
		K_subspace = one_minus_P.dot(self.noise_matrix.todense().dot(one_minus_P))
		temp =  A_subspace.dot(C_subspace.dot(A_subspace.T.conj()) - A_subspace.dot(C_subspace))


	def plot_entropy(self, label):
		plt.subplot(2, 1, 1)
		plt.plot(np.real(self.entropy))
		plt.subplot(2, 1, 2)
		plt.plot(self.final_phi)
		plt.title("total entropy production = {}".format(np.sum(self.entropy)))
		plt.savefig("{}_entropy.pdf".format(label))
		plt.close()

	def write_entropy(self, label):
		np.save("{}_entropy.npy".format(label), self.entropy)

	def read_entropy(self, label):
		self.entropy = np.load("{}_entropy.npy".format(label))

	def small_amp_expansion(self):
		size = 10
		assert np.abs(self.phi_target) < 1e-10
		A1 = - 2 * self.u
		epsilon = self.a**2 /(4 * self.k) - self.u*self.phi_shift
		k_c = np.sqrt(self.a/(2 * self.k))
		def A0(k):
			return self.a * k**2 - self.k * k**4 - self.u*(self.phi_shift)
		def K(k):
			return 2 * k**2 + self.u * self.phi_shift
		ks = 2 * np.pi * np.arange(size)/self.X
		S = np.zeros(size)
		for (i, k) in enumerate(ks):
			f1 = (1 - K(k+k_c)/K(k))/(A0(k) + A0(k+k_c))
			print(f1)
			f2 = (1 - K(k-k_c)/K(k))/(A0(k) + A0(k-k_c))
			S[i] = epsilon * A1**2 * (f1 + f2)

		plt.plot(S)
		plt.show()

	def _make_correlation_matrix(self):
		self.correlation_matrix = sl.solve_continuous_lyapunov(self.first_order_matrix, (-self.noise_matrix).todense())

		# Check accuracy of the Lyapunov eq
		temp = self.first_order_matrix.dot(self.correlation_matrix) + self.correlation_matrix.dot(self.first_order_matrix.T.conj())
		print("Error in Lyapunov eq: ", sl.norm(temp + self.noise_matrix.todense())/sp.linalg.norm(self.noise_matrix))
		print("Norm of A: ", sl.norm(self.first_order_matrix))
		print("Norm of C: ", sl.norm(self.correlation_matrix))
		print("Norm of K: ", sp.linalg.norm(self.noise_matrix))

		# Project into subspace and check accuracy
		identity = np.identity(self.size)
		one_minus_P = identity - self.projection

		C_subspace = one_minus_P.dot(self.correlation_matrix.dot(one_minus_P))
		A_subspace = one_minus_P.dot(self.first_order_matrix.dot(one_minus_P))
		K_subspace = one_minus_P.dot(self.noise_matrix.todense().dot(one_minus_P))
		temp = A_subspace.dot(C_subspace) + C_subspace.dot(A_subspace.T.conj())
		print("Error in Lyapunov eq in subspace: ", sl.norm(temp + K_subspace)/sl.norm(K_subspace))

	def _make_first_order_matrix(self):
		temp_diag = self.a * (3 * self.final_phi**2 - 1)
		temp = sp.diags([temp_diag], [0], shape=(self.size, self.size))
		self.first_order_matrix_orig = self._laplacian_sparse * temp - self.k * self._laplacian_sparse * self._laplacian_sparse
		temp_diag = self.u
		self.first_order_matrix_orig -= sp.diags([temp_diag], [0], shape=(self.size, self.size))
		self.first_order_matrix = self.first_order_matrix_orig.todense()

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
		self.projection = np.outer(goldstone_mode, goldstone_mode.conj())
		self.first_order_matrix -= self.projection * reg

	def _make_noise_matrix(self):
		self.noise_matrix = -2*self._laplacian_sparse + sp.diags([self.u], [0], shape=(self.size, self.size))

	def _make_laplacian_matrix(self):
		diags = np.array([1, 1, -2, 1, 1])/self.dx**2
		self._laplacian_sparse = sp.diags(diags, [-self.size+1, -1, 0, 1, self.size-1], shape=(self.size, self.size))

	def _inverse(self, sparse_matrix):
		inverse = sp.linalg.inv(sparse_matrix)
		identity = sp.identity(sparse_matrix.shape[0])
		print("error of inverse: {}".format(sp.linalg.norm(inverse * sparse_matrix - identity)))
		return inverse

	def _multiply_in_fourier_space(self):
		temp = (self.first_order_matrix_orig @ (self.correlation_matrix @ self.first_order_matrix_orig.T)).todense()
		temp = fft(temp).T.conj()
		temp = fft(temp).T.conj()/self.size
		jac_f = fft(self.first_order_matrix_orig.todense()).T.conj()
		jac_f = fft(jac_f).T.conj()/self.size
		x = np.arange(self.size)
		k_f_diag = 1/(4 * (1 - np.cos(2 * np.pi * x/self.size)) + self.u)

		S_f = np.einsum('ij,j->ij', temp, k_f_diag)
		print("S in fourier: ", 2 * np.trace(S_f) + self.first_order_matrix_orig.diagonal().sum())
		entropy_f = 2 * S_f.diagonal() + jac_f.diagonal()
		print(entropy_f)
		plt.plot(S_f.diagonal())
		plt.plot(jac_f.diagonal())
		plt.plot(entropy_f)
		plt.show()


		# Back to real space
		S = ifft(S_f).T.conj()
		S = np.real(ifft(S).T.conj() * self.size)
		S = 2 * S + self.first_order_matrix_orig.todense()
		print("S in real: ", np.trace(S))
		plt.plot(np.diag(S))
		plt.show()

		return np.diag(S)


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
		self.final_phi = self.phi[-2]

		self._make_laplacian_matrix()
		self._make_first_order_matrix()
		self._make_noise_matrix()
		self._add_to_translational_dof(reg=reg)
		self._make_correlation_matrix()

		self.correlation_matrix = sp.csr_matrix(self.correlation_matrix)
		S = self.first_order_matrix_orig @ (self.correlation_matrix @ self.first_order_matrix_orig.T.conj())
		S = 2 * np.einsum('ij, j->ij', S.todense(), 1/self.noise_matrix.diagonal()) + self.first_order_matrix_orig.todense()

		S_real = self._ifft_matrix(S)
		self.entropy = S_real.diagonal()

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


	def _fft_matrix(self, matrix):
		matrix_fft = fft(matrix).T.conj()
		matrix_fft = fft(matrix_fft).T.conj()/self.size
		return matrix_fft

	def _ifft_matrix(self, matrix):
		matrix_ifft = ifft(matrix).T.conj()
		matrix_ifft = ifft(matrix_ifft).T.conj() * self.size
		return matrix_ifft




if __name__ == "__main__":

	label = 'u_1e-06'

	solver = EntropyProductionFourier()
	solver.load(label)
	solver.test()
	# solver.calculate_entropy_quad_bd()
	# solver.small_amp_expansion()
	# solver.plot_entropy(label)
