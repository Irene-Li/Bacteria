import numpy as np
import time 
from FdEvolution import FdEvolution
from matplotlib import pyplot as plt



class AltEvolution(FdEvolution):

	def initialise(self, X, dx, T, dt, n_batches, initial_value, double=False):
		self.dx = dx 
		self.size = int(X/dx) + 5 + 4 
		self.X = X 
		self.T = T 
		self.dt = dt 
		self.n_batches = int(n_batches)
		self.step_size = T/self.n_batches
		self.batch_size = int(self.step_size/self.dt)
		self._modify_params()
		
		if double:
			self.phi_initial = self._double_tanh(initial_value)
		else: 
			self.phi_initial = self._slow_mfold(initial_value)	


	def _enforce_bc(self, phi):
		phi[0:4] = phi[7:3:-1]
		phi[-4:] = phi[-6:-10:-1]
		return phi 

	def _laplacian(self, phi):
		phi_left = np.roll(phi, -2)
		phi_right = np.roll(phi, 2)
		return (phi_right + phi_left - 2 * phi)/(4 * (self.dx)**2)



	def _sin_surface(self, phi_average):
		x = np.arange(-4 * self.dx, self.dx * (self.size-4), self.dx)
		phi_initial = self._init_sin(x, phi_average)
		self.phi_initial = self._enforce_bc(phi_initial)


	def _random_init(self, phi_average):
		phi_initial = phi_average + np.random.normal(0, 0.01, self.size)
		self.phi_initial = self._enforce_bc(phi_initial)

	def comparison(self):
		odd_points = self.phi[:, ::2] 
		even_points = self.phi[:, 1:-1:2]
		average = ((np.roll(odd_points, 1) + odd_points)/2)[:, 1:]
		diff = np.mean(np.abs(average - even_points), axis=-1)
		plt.plot(diff)
		plt.show()


if __name__ == '__main__':
	label = 'test'

	# parameters of the differential equation
	a = 1e-1
	k = 1

	# simulation parameters
	X = 60
	dx = 0.2
	dt = 1e-3
	T = 1e5
	n_batches = 100

	# Evolve
	u = 0 
	phi_shift = 1.1
	phi_target = 0 
	phi_init = 0 
	double = False 

	n_samples = 3 

	start_time = time.time()
	solver = AltEvolution(a, k, u, phi_target, phi_shift)
	solver.initialise(X, dx, T, dt, n_batches, phi_init, double=double)
	solver._sin_surface(phi_init)
	solver.evolve()
	solver.save(label)
	end_time = time.time()
	print('The simulation took: ')
	print(end_time - start_time)




