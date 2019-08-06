import numpy as np
from StoEvolution1D import *

X= 400
inits = ['flat', 'tanh']
u = 5e-6

label = 'X_{}_u_{}_flat'.format(X, u)
solver = StoEvolution1D()
solver.load(label)
solver.plot_evolution(label, t_size=100, x_size=200)
