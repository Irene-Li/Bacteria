import numpy as np
from StoEvolutionPS import *

labels = ['phi_t_-0.8_nuc']
t_size = 100
x_size = 200
u = 1e-5


for phi_t in [-0.6, -0.7]:
	label = 'phi_t_{}_skewed_droplet'.format(phi_t)
	solver = StoEvolutionPS()
	solver.load(label)
	# solver.print_params()
	solver.make_movie(label)
