import numpy as np
from StoEvolutionPS import *

labels = ['phi_t_-0.8_nuc']
t_size = 100
x_size = 200
u = 1e-5


for u in [1e-5, 5e-5]:
	label = 'u_{}_skewed_droplet'.format(u)
	solver = StoEvolutionPS()
	solver.load(label)
	# solver.print_params()
	solver.make_movie(label)
