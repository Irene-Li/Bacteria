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

		C_subspace = self._project_matrix(self.correlation_matrix)
		A_subspace = self._project_matrix(self.first_order_matrix_orig.todense())
		K_subspace = self._project_matrix(self.noise_matrix.todense())
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
		self.projection_onto_gm = np.outer(goldstone_mode, goldstone_mode.conj())
		self.first_order_matrix -= self.projection_onto_gm * reg
		self.projection_out_gm = np.identity(self.size) - self.projection_onto_gm

	def _project_matrix(self, matrix):
		return self.projection_out_gm.dot(matrix.dot(self.projection_out_gm))



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

		self._make_laplacian_matrix()
		self._make_first_order_matrix()
		self._make_noise_matrix()
		self._add_to_translational_dof(reg=reg)
		self._make_correlation_matrix()


		S = self._calculate_entropy_with_conjugate_currents()
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

	def entropy_with_modelAB_currents(self):
		self._make_laplacian_matrix()
		self._make_gradient_matrix()
		final_phi_fourier = fft(self.final_phi)
		final_phi_cube_fourier = fft(self.final_phi**3)
		mu = self.a*(-final_phi_fourier + final_phi_cube_fourier) - self.k*self._laplacian_fourier*final_phi_fourier
		J_1 = self._gradient_fourier * mu
		J_1 = ifft(J_1)

		J_2 = self.u*(self.final_phi + self.phi_shift)*(self.final_phi - self.phi_target)
		M_2 = self.u*(self.phi_shift+self.phi_target)/2

		self.entropy_from_model_B_current = J_1*J_1
		self.entropy_from_model_A_current = J_2*J_2/M_2
		self.entropy = self.entropy_from_model_A_current + self.entropy_from_model_B_current

	def plot_entropy_from_modelAB_currents(self, label):

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif', size=12)

		plt.subplot(2, 1, 1)
		plt.plot(np.real(self.entropy), 'k-', label="total entropy")
		plt.plot(np.real(self.entropy_from_model_A_current), 'c-', label='model A entropy')
		plt.plot(np.real(self.entropy_from_model_B_current), 'b-', label='model B entropy')
		plt.title(r"The spatial decomposition of the entropy production")
		plt.ylabel(r"$\dot{S}$")
		plt.subplot(2, 1, 2)
		plt.plot(self.final_phi, 'k-')
		plt.ylabel(r"$\phi$")
		plt.xlabel(r"$x$")
		plt.savefig("{}_entropy_modelAB_current.pdf".format(label))
		plt.close()

	def _calculate_entropy_with_jac(self):
		C = sp.csr_matrix(self.correlation_matrix)
		A = self.first_order_matrix_orig
		S = A @ (C @ A.T.conj())
		S = 2 * np.einsum('ij, j->ij', S.todense(), 1/self.noise_matrix.diagonal()) + A
		return S


	def _calculate_entropy_with_A_tilde(self):
		self._make_A_tilde()
		S = self.first_order_matrix_orig.todense().dot(self.correlation_matrix.dot(self.A_tilde.T.conj()))
		S = 2 * np.einsum('ij, j->ij', S, 1/self.noise_matrix.diagonal()) + self.A_tilde
		return S

	def _calculate_entropy_with_antisym_A(self):
		# Decompose A into sym and antisymmetric parts
		C = self._project_matrix(self.correlation_matrix)
		A = self.first_order_matrix_orig.todense()
		K_diag = self.noise_matrix.diagonal()
		B = np.einsum('i, ij->ij', 1/K_diag, A)
		B_antisym = (B - B.T.conj())/2

		sqrt_K = np.sqrt(K_diag)
		sqrt_K_B = np.einsum('i, ij->ij', sqrt_K, B)
		sqrt_K_B_antisym = np.einsum('ij, j->ij', B_antisym, sqrt_K)

		term1 = - 2*sqrt_K_B.dot(C.dot(sqrt_K_B_antisym))
		term2 = np.einsum('i, ij->ij', sqrt_K, sqrt_K_B_antisym)

		S = term1 + term2

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

		plt.plot(np.einsum('ij,j->i', E, self.final_phi))
		plt.show()

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
		K_dot_C_inv = np.einsum('i,ij->ij', K_diag, C_inv)

		cross_term = K_dot_C_inv.dot(C.dot(E.T.conj()))
		print(np.trace(cross_term))
		S = K_dot_E.dot(C.dot(E.T.conj()))
		return S

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
		diag = -2*self._laplacian_fourier + self.u*(self.phi_shift+self.phi_target)
		self.noise_matrix = sp.diags([diag], [0], shape=(self.size, self.size))

	def _make_noise_matrix_lin_bd(self):
		diag = -2*self._laplacian_fourier + self.u
		self.noise_matrix = sp.diags([diag], [0], shape=(self.size, self.size))










if __name__ == "__main__":

	label = 'X_50_u_1e-6'

	solver = EntropyProductionFourier()
	solver.load(label)

	# solver.read_entropy(label)

	# solver.compare_entropy()
	# solver.entropy_with_modelAB_currents()
	# solver.plot_entropy_from_modelAB_currents(label)

	solver.calculate_entropy()
	solver.plot_entropy(label)
