import numpy as np
import matplotlib.pyplot as plt
from EvolveModelAB import *

# Run parameters
M1 = 0.2
k = 5
X = 100
dx = 1
dt = 1e-3
n_batches = 100
flat = True

M2 = 5e-6
T = 10/M2
reg = 5

for delta_b in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    # Label for the run
    label = 'M2_{}_delta_b_{}'.format(M2, delta_b)
    print(label)

    # start_time = time.time()
    # solver = EvolveModelAB(M1, k, M2, delta_b)
    # solver.initialise(X, dx, T, dt, n_batches, 0, flat=flat)
    # solver.evolve()
    # solver.save(label)
    # end_time = time.time()
    # print('The simulation took: ')
    # print(end_time - start_time)
    #
    # solver = EntropyModelAB()
    # solver.load(label)

    # solver.read_entropy(label)
    solver.entropy_with_modelAB(current=False, reg=reg)
    solver.plot_entropy(label+'_phi')
    solver.write_entropy(label+'_phi')
    solver.entropy_with_modelAB(current=True)
    solver.plot_entropy(label+'_current')
    solver.write_entropy(label+'_current')
