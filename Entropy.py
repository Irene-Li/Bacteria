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

	def calculate_entropy(self):
		reg = 1

		self._make_laplacian_matrix()
		self._make_first_order_matrix()
		self._make_noise_matrix()
		self._add_to_translational_dof(reg=reg)
		self._make_correlation_matrix()

		self.correlation_matrix = sp.csr_matrix(self.correlation_matrix)
		self.entropy = self._multiply_in_fourier_space()

	def load(self, label):
		super(EntropyProduction, self).load(label)
		self.final_phi = self.phi[-2]


	def test(self):
		reg = 1
		self.correlation_matrix = np.load("correlation.npy")
		self._make_laplacian_matrix()
		self._make_first_order_matrix()
		self._make_noise_matrix()
		self._add_to_translational_dof(reg=reg)

		C_subspace = self.projection_out_gm.dot(self.correlation_matrix.dot(self.projection_out_gm))
		A_subspace = self.projection_out_gm.dot(self.first_order_matrix.dot(self.projection_out_gm))
		K_subspace = self.projection_out_gm.dot(self.noise_matrix.todense().dot(self.projection_out_gm))
		temp =  A_subspace.dot(C_subspace.dot(A_subspace.T.conj()) - A_subspace.dot(C_subspace))


	def plot_entropy(self, label):
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif', size=12)

		plt.subplot(2, 1, 1)
		plt.plot(np.real(self.entropy), 'k-')
		plt.title(r"The spatial decomposition of the entropy production")
		plt.ylabel(r"$\dot{S}$")
		plt.subplot(2, 1, 2)
		plt.plot(self.final_phi, 'k-')
		plt.ylabel(r"$\phi$")
		plt.xlabel(r"$x$")
		plt.savefig("{}_entropy.pdf".format(label))
		plt.close()

	def write_entropy(self, label):
		np.save("{}_entropy.npy".format(label), self.entropy)

	def read_entropy(self, label):
		filename = "{}_entropy.npy".format(label)
		if os.path.isfile(filename):
			self.entropy = np.load(filename)
		else:
			self.calculate_entropy()
			self.write_entropy(label)


	def _make_correlation_matrix(self):
		self.correlation_matrix = sl.solve_lyapunov(self.first_order_matrix, (-self.noise_matrix).todense())

		# Check accuracy of the Lyapunov eq
		temp = self.first_order_matrix.dot(self.correlation_matrix) + self.correlation_matrix.dot(self.first_order_matrix.T.conj())
		print("Error in Lyapunov eq: ", sl.norm(temp + self.noise_matrix.todense())/sp.linalg.norm(self.noise_matrix))
		print("Norm of A: ", sl.norm(self.first_order_matrix))
		print("Norm of C: ", sl.norm(self.correlation_matrix))
		print("Norm of K: ", sp.linalg.norm(self.noise_matrix))

		self.correlation_matrix = self.projection_out_gm.dot(self.correlation_matrix.dot(self.projection_out_gm))
		A_subspace = self.projection_out_gm.dot(self.first_order_matrix.dot(self.projection_out_gm))
		K_subspace = self.projection_out_gm.dot(self.noise_matrix.todense().dot(self.projection_out_gm))
		temp = A_subspace.dot(self.correlation_matrix) + self.correlation_matrix.dot(A_subspace.T.conj())
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
		projection_onto_gm = np.outer(goldstone_mode, goldstone_mode.conj())
		self.first_order_matrix -= projection_onto_gm * reg
		self.projection_out_gm = np.identity(self.size) - projection_onto_gm

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

	def calculate_entropy(self, with_A_tilde=False):
		reg = 1

		self._make_laplacian_matrix()
		self._make_first_order_matrix()
		self._make_noise_matrix()
		self._add_to_translational_dof(reg=reg)
		self._make_correlation_matrix()

		self.correlation_matrix = sp.csr_matrix(self.correlation_matrix)
		if with_A_tilde:
			S = self.u * self._calculate_entropy_with_A_tilde()
		else:
			S = self._calculate_entropy_with_jac()
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

		S1 = self._calculate_entropy_with_A_tilde_2()
		S2 = self._calculate_entropy_with_jac()
		print("total entropy production of S1:", np.trace(S1))
		print("total entropy production of S2:", np.trace(S2))
		S1_real = self._ifft_matrix(S1)
		S2_real = self._ifft_matrix(S2)
		plt.plot((np.diag(S1)-np.diag(S2))[:10], 'ko--', label='diff')
		plt.legend()
		plt.show()


	def _calculate_entropy_with_jac(self):
		self.correlation_matrix = sp.csr_matrix(self.correlation_matrix)
		S = self.first_order_matrix_orig @ (self.correlation_matrix @ self.first_order_matrix_orig.T.conj())
		S = 2 * np.einsum('ij, j->ij', S.todense(), 1/self.noise_matrix.diagonal()) + self.first_order_matrix_orig.todense()
		return S


	def _calculate_entropy_with_A_tilde(self):
		self._make_A_tilde()
		S = self.first_order_matrix_orig.todense().dot(self.correlation_matrix.dot(self.A_tilde.T.conj()))
		S = 2 * np.einsum('ij, j->ij', S, 1/self.noise_matrix.diagonal()) + self.A_tilde
		return S

	def _calculate_entropy_circ_1(self):
		self._make_A_tilde()
		S = self.correlation_matrix.dot(self.A_tilde.T.conj())
		S = np.einsum('ij, j->ij', S, 1/self.noise_matrix.diagonal())
		S = np.matmul(S, self.first_order_matrix_orig.todense())
		S = 2 * S + self.A_tilde
		return S

	def _calculate_entropy_with_antisym_A(self):
		# Decompose A into sym and antisymmetric parts
		K_inv_A = np.einsum('ij, i->ij', self.first_order_matrix_orig.todense(), 1/self.noise_matrix.diagonal())
		K_inv_A_antisym = (K_inv_A - K_inv_A.T.conj())/2

		S = self.first_order_matrix_orig.todense().dot(self.correlation_matrix.dot(K_inv_A_antisym.T.conj()))
		S = 2 * S + np.einsum('i, ij->ij', self.noise_matrix.diagonal(), K_inv_A_antisym)
		return S

	def _calculate_entropy_with_A_tilde_2(self):
		K_inv_A = np.einsum('i, ij->ij', 1/self.noise_matrix.diagonal(), self.first_order_matrix_orig.todense())
		mu_0 =  (3/2 * self.a) * self._fft_matrix(np.diag(self.final_phi**2))
		diag = (- self.a/2 - self.k/2 * self._laplacian_fourier)
		mu_0 += sp.diags([diag], [0], shape=(self.size, self.size)).todense()
		K_inv_A_tilde = self._project_matrix(K_inv_A - mu_0)
		A_tilde = np.einsum('i, ij->ij', self.noise_matrix.diagonal(), K_inv_A_tilde)

		# A_tilde = self._project_matrix(A_tilde)
		# K_inv_A_tilde = np.einsum('i, ij->ij', 1/self.noise_matrix.diagonal(), A_tilde)

		S = self.first_order_matrix_orig.todense().dot(self.correlation_matrix.dot(K_inv_A_tilde.T.conj()))
		S = 2 * S + A_tilde
		return S

	def _project_matrix(self, matrix):
		return self.projection_out_gm.dot(matrix.dot(self.projection_out_gm))

	def _make_A_tilde(self):
		prefactor = (self.phi_shift + self.phi_target)/2
		A_tilde = 3 * self.a * prefactor * self._fft_matrix(np.diag(self.final_phi**2))
		diag = prefactor*(- self.a - self.k * self._laplacian_fourier) - (self.phi_shift - self.phi_target)
		A_tilde += sp.diags([diag], [0], shape=(self.size, self.size)).todense()
		A = self._fft_matrix(np.diag(2 * self.final_phi))
		self.A_tilde = A_tilde - A



	def small_param_expansion(self):
		self._make_laplacian_matrix()
		f = self.u * fft((self.final_phi + self.phi_shift)*(self.final_phi - self.phi_target))
		f /= ( - 2 * self._laplacian_fourier + self.u * (self.phi_shift + self.phi_target))
		plt.plot(ifft(f))
		plt.show()


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

	def _make_noise_matrix_quad_bd(self):
		diag = -2*self._laplacian_fourier + self.u*(self.phi_shift+self.phi_target)
		self.noise_matrix = sp.diags([diag], [0], shape=(self.size, self.size))

	def _make_noise_matrix_lin_bd(self):
		diag = -2*self._laplacian_fourier + self.u
		self.noise_matrix = sp.diags([diag], [0], shape=(self.size, self.size))









if __name__ == "__main__":

	label = 'medium_box_u_1e-6_2'

	solver = EntropyProductionFourier()
	solver.load(label)

	# solver.read_entropy(label)
	# solver.small_param_expansion()
	solver.compare_entropy()

	# solver.plot_entropy(label)
