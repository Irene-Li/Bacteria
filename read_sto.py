import numpy as np
from StoEvolution import *

labels = ['sto_ep_0.1_flat_long', 'sto_ep_0.1_sin_long']
t_size = 100
x_size = 200


for label in labels:
	solver = StoEvolution()
	solver.load(label)
	solver.rescale_to_standard()
	solver.load_phi(label)
	solver.plot_evolution(t_size, x_size,label)
	solver.plot_steady_state(label)
