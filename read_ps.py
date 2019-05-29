import time
import numpy as np
from PsEvolution import *

labels = ['u_1e-05_dt_0.001']
t_size = 100
x_size = 200

for label in labels:
	solver = PsEvolution()
	solver.load(label)
	solver.make_movie(label)
