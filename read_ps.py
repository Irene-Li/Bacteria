import time
import numpy as np
from PsEvolution import *

labels = ['u_0_dt_0.0001']
t_size = 100
x_size = 200

for u in [1e-4, 1e-5]:
	label = 'u_{}_skewed_droplet'.format(u)
	solver = PsEvolution()
	solver.load(label)
	# solver.print_params()
	solver.make_movie(label)
