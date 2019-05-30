import numpy as np
from StoEvolutionPS import *

labels = ['ep_0_flat_False']
t_size = 100
x_size = 200
u = 1e-5


for phi_shift in [10]:
	u = 1e-4/phi_shift
	label = 'u_{}_phi_s_{}'.format(u, phi_shift)
	solver = StoEvolutionPS()
	solver.load(label)
	solver.make_movie(label)
