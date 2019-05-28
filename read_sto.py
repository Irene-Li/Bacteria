import numpy as np
from StoEvolutionPS import *

labels = ['sto_ep_0.01_test']
t_size = 100
x_size = 200


for label in labels:
	solver = StoEvolutionPS()
	solver.load(label)
	solver.rescale_to_standard()
	solver.load_phi(label)
	solver.make_movie(label)
