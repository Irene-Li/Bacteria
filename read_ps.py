import time
import numpy as np
from PsEvolution import *

labels = ['u_0_dt_0.0001']
t_size = 100
x_size = 200

for u in [1e-5, 1e-6]:
    for dt in [5e-3]:
        label = 'u_{}_dt_{}'.format(u, dt)
        solver = PsEvolution()
        solver.load(label)
        solver.make_movie(label)
