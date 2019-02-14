import numpy as np
import matplotlib.pyplot as plt
import time
import dolfin as df
from PDE import * 


class FeEvolution(TimeEvolution):

	def initialise(self, X, dx, T, dt, n_batches, initial_value, theta=0):
		self.dx = dx 
		self.X = X 
		self.size = int(X/dx) + 1 
		self.T = T 
		self.dt = dt 
		self.n_batches = int(n_batches)
		self.batch_size = int((self.T/self.dt)/self.n_batches)
		self.step_size = self.T/self.n_batches 
		self._modify_params()
		self.phi_initial = initial_value
		self.theta = theta 

	def evolve(self):
		u_init = self._set_initial_conditions()
		self._set_up_problem()

		self.u_curr.interpolate(u_init)
		self.u_prev.interpolate(u_init)

		solver = df.NewtonSolver()
		solver.parameters["linear_solver"] = "lu"
		solver.parameters["convergence_criterion"] = "incremental"
		solver.parameters["relative_tolerance"] = 1e-8
		solver.parameters["report"] = False 

		self.phi = np.zeros((self.n_batches, self.size))

		n = 0 
		for i in range(int(self.T/self.dt)):
		    if i % self.batch_size == 0:
		    	self.phi[n] = self.u_curr.split()[0].compute_vertex_values()
		    	n += 1 
		    self.u_prev.vector()[:] = self.u_curr.vector()
		    solver.solve(self.problem, self.u_curr.vector())	   
		    
	def _modify_params(self):
		# modify parameters 
		length_ratio = 1/(self.X)
		self.dx = self.dx * length_ratio
		self.X = 1 
		self.k = self.k * (10/self.a) * length_ratio**2 
		time_ratio = length_ratio**2 * (self.a/10)
		self.u = self.u/time_ratio 
		self.a = 10 
		self.b = 10 
		self.dt = self.dt * time_ratio
		self.T = self.T * time_ratio
		self.step_size = self.step_size * time_ratio 


	def _set_initial_conditions(self):
		phi_b = np.sqrt(self.a/self.b)
		f = lambda x: self._init_sin(x, self.phi_initial)
		class InitialConditions(df.UserExpression):
			def eval(self, values, x):
				values[0] = f(x)
				values[1] = 0.0 
			def value_shape(self):
				return (2, )

		return InitialConditions(degree=1)

	def _set_up_problem(self):
		self.mesh = df.UnitIntervalMesh(self.size-1)
		print(np.array(self.mesh.coordinates()).shape)

		P1 = df.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
		ME = df.FunctionSpace(self.mesh, P1*P1)

		du = df.TrialFunction(ME)
		q, v = df.TestFunctions(ME)

		u_curr = df.Function(ME)
		u_prev = df.Function(ME)

		dphi, dmu = df.split(du)
		phi_curr, mu_curr = df.split(u_curr)
		phi_prev, mu_prev = df.split(u_prev)

		mu_mid = (1.0 - self.theta)*mu_prev + self.theta*mu_curr 
		phi_mid = (1.0 - self.theta)*phi_prev + self.theta*phi_curr 

		dfdphi =  - self.a * phi_curr + self.b * phi_curr**3 

		dx = df.dx 
		inner = df.inner 
		grad = df.grad 

		L1 = phi_curr*q*dx - phi_prev*q*dx + self.dt*inner(grad(mu_mid), grad(q))*dx 
		if self.u > df.DOLFIN_EPS:
			birth_death = self.u*(phi_mid + self.phi_shift)*(phi_mid - self.phi_target)
			L1 += self.dt*birth_death*q*dx 
		L2 = mu_curr*v*dx - dfdphi*v*dx - self.k*inner(grad(phi_curr), grad(v))*dx
		linear = L1 + L2 
		bilinear = df.derivative(linear, u_curr, du)

		self.problem = self.CustomProblem(bilinear, linear)
		self.u_curr = u_curr 
		self.u_prev = u_prev 


	class CustomProblem(df.NonlinearProblem):

		def __init__(self, bilinear, linear):
			df.NonlinearProblem.__init__(self)
			self.bilinear = bilinear 
			self.linear = linear 

		def F(self, residual, x):
			df.assemble(self.linear, tensor=residual)

		def J(self, jacobian, x):
			df.assemble(self.bilinear, tensor=jacobian)