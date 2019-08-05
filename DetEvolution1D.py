import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import ode
import scipy.sparse as sp
import json
from TimeEvolution import TimeEvolution
from pseudospectral import delta_phi_ps_1d


class DetEvolution1D(TimeEvolution):

	def evolve(self, verbose=False, ps=True):
		if ps:
			self.evolve_ps(verbose)
		else:
			self.evolve_fd(verbose)


	def evolve_fd(self, verbose=False):
		self.phi  = np.zeros((self.n_batches, self.size))
		self._make_laplacian_matrix()

		small_batch = self.batch_size
		while small_batch > 1000:
			small_batch /= 10 # decrease the amount of time integration at each step

		r = ode(self._fd_delta_phi).set_integrator('lsoda', atol=1e-10, nsteps=small_batch)
		r.set_initial_value(self.phi_initial, 0)

		n = 0
		phi = self.phi_initial

		for i in range(int((self.T/self.dt)/small_batch)):
			if r.successful():
				if i % int(self.batch_size/small_batch) == 0:
					self.phi[n] = phi
					if verbose:
						print('iteration: {}	mean: {}'.format(n, np.mean(phi)))
					n += 1
				phi = r.integrate(r.t+self.dt*small_batch)

	def evolve_ps(self, verbose=True):
		self.phi  = np.zeros((self.n_batches, self.size))
		self.kmax_half = np.pi/2
		self.kmax_two_thirds = np.pi*2/3
		self.factor = np.pi*2/self.size

		small_batch = self.batch_size
		while small_batch > 1000:
			small_batch /= 10 # decrease the amount of time integration at each step

		r = ode(self._ps_delta_phi).set_integrator('lsoda', atol=1e-8, nsteps=small_batch)
		r.set_initial_value(self.phi_initial, 0)

		n = 0
		phi = self.phi_initial

		for i in range(int((self.T/self.dt)/small_batch)):
			if r.successful():
				if i % int(self.batch_size/small_batch) == 0:
					self.phi[n] = phi
					if verbose:
						print('iteration: {}	mean: {}'.format(n, np.mean(phi)))
					n += 1
				phi = r.integrate(r.t+self.dt*small_batch)

	def _ps_delta_phi(self, t, phi):
		phi_complex = phi.astype('complex128')
		delta_phi = delta_phi_ps_1d(phi_complex, self.a, self.k, self.u,
									 self.phi_shift, self.phi_target,
									 self.kmax_half, self.kmax_two_thirds,
									 self.factor, self.size)
		return np.real(delta_phi)

	def rescale_to_standard(self):
		time_ratio = self.a**2/self.k
		space_ratio = np.sqrt(self.a/self.k)
		self.dx = space_ratio * self.dx
		self.X = space_ratio * self.X
		self.T = time_ratio * self.T
		self.dt = time_ratio * self.dt
		self.step_size = time_ratio * self.step_size
		self.a = 1
		self.k = 1
		self.u = self.u/time_ratio


	def _diff(self, phi):
		phi_left = np.roll(phi, -1)
		phi_right = np.roll(phi, 1)
		return (phi_right - phi_left)/(self.dx * 2)

	def _laplacian(self, phi):
		return self._laplacian_sparse.dot(phi)

	def _enforce_bc(self, phi):
		# phi[1] = phi[3]
		# phi[0] = phi[4]
		# phi[-2] = phi[-4]
		# phi[-1] = phi[-5]

		return phi


	def _fd_delta_phi(self, t, phi):

		mu = self.a * ( - phi + phi ** 3) - self.k * self._laplacian(phi)
		lap_mu = self._laplacian(mu)
		delta = lap_mu - self.u * (phi + self.phi_shift) * (phi - self.phi_target)
		return delta


	def _make_laplacian_matrix(self):
		diags = np.array([1, 1, -2, 1, 1])/self.dx**2
		self._laplacian_sparse = sp.diags(diags, [-self.size+1, -1, 0, 1, self.size-1], shape=(self.size, self.size))
		self._laplacian_dense = self._laplacian_sparse.todense()


	def _fd_jac(self, t, phi):
		temp = sp.diags([self.a *(3 * phi**2 - 1)], [0], shape=(self.size, self.size))
		jac = self._laplacian_sparse * temp - self.k * self._laplacian_sparse * self._laplacian_sparse
		temp_diag = self.u * ( 2 * phi + self.phi_shift - self.phi_target)
		jac -= sp.diags([temp_diag], [0], shape=(self.size, self.size))
		return jac.todense()

	def _average_for_nbc(self, phi):
		return super()._average(phi[:, 2:-2])

	def _average_vector_for_nbc(self, phi):
		return super()._average_vector(phi[2:-2])

	def _compute_f(self):
		dphi = self._diff(self.phi)
		f = - self.a/2 * self.phi**2 + self.a/4 * self.phi**4 + self.k/2 * dphi **2
		return self._average_for_nbc(f)


	def compute_mu(self, label):

		x = np.arange(0, self.size * self.dx, self.dx)

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		phi = self.phi[-2]
		mu = - self.a * phi + self.a * (phi ** 3) - self.k * self._laplacian(phi)
		plt.plot(x[2:-2], mu[2:-2])
		plt.title(r'$\mu = \alpha (- \phi + \phi^3) - \kappa \partial_x^2 \phi$')
		plt.xlabel(r'$x$')
		plt.ylabel(r'$\mu$')
		plt.tight_layout()
		plt.savefig('{}_mu.pdf'.format(label))
		plt.close()
