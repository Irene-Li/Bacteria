import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import ode 


class PsEvolution(TimeEvolution):
	# this doesn't really work 

	def evolve(self):
		# Set up problem 
		from dedalus import public as de

		xbasis = de.Chebyshev('x', self.size, interval=(0, self.size * self.dx), 
								dealias=3/2)

		domain = de.Domain([xbasis], grid_dtype=np.float64)
		problem = de.IVP(domain, variables=['phi', 'phi_x', 'phi_xx', 'mu', 'mu_x'])

		problem.meta['phi_x']['x']['dirichlet'] = True
		problem.meta['mu_x']['x']['dirichlet'] = True


		# Set parameters 
		problem.parameters['a'] = self.a
		problem.parameters['b'] = self.b
		problem.parameters['k'] = self.k 
		problem.parameters['u'] = self.u
		problem.parameters['phi_t'] = self.phi_target
		problem.parameters['phi_0'] = self.phi_shift 

		# Add equations
		problem.add_equation("dt(phi) - dx(mu_x) = 0")
		problem.add_equation("mu_x - dx(mu) = 0")
		problem.add_equation("mu + a*phi + k*phi_xx = b*phi**3")
		problem.add_equation("phi_xx - dx(phi_x) = 0")
		problem.add_equation("phi_x - dx(phi) = 0")


		# Boundary conditions 
		problem.add_bc('right(phi_x) = 0')
		problem.add_bc('left(phi_x) = 0')
		problem.add_bc('left(mu_x) = 0')
		problem.add_bc('right(mu_x) = 0')


		solver = problem.build_solver(de.timesteppers.RK222)

		# Set up initial conditions 
		x = domain.grid(0)
		phi_variable = solver.state['phi']
		phi_variable['g'] = self.phi_initial
		

		# Set time 
		solver.stop_wall_time = np.inf
		solver.stop_sim_time = np.inf 
		solver.stop_iteration = int(self.T/self.dt)

		phi_variable.set_scales(1, keep_data=True)
		self.phi = np.zeros((self.n_batches, self.size))

		n = 0
		while solver.ok:
		    solver.step(self.dt)
		    phi_variable.set_scales(1, keep_data=True)

		    if solver.iteration % self.batch_size == 0:
			    self.phi[n] = np.copy(phi_variable['g'])
			    print('Iteration: {}, mean: {}'.format(solver.iteration, np.mean(self.phi[n])))
			    n += 1



