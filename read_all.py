import time
import numpy as np
from Entropy import *

rates = [1e-5, 5e-6, 5e-5, 1e-4, 2.5e-6, 2.5e-5]  

for u in rates:
	label = 'X_200_u_{}'.format(u)
	solver = EntropyProduction()
	solver.load(label)
	solver.calculate_entropy_quad_bd()
	solver.plot_entropy(label)