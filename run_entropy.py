import numpy as np
import matplotlib.pyplot as plt
from EvolveModelAB import *
M2 = 1e-6
delta_b = 0.1
reg = 5

for delta_b in np.arange(0.2, 1, 0.1):
    label = 'M2_{}_delta_b_{}'.format(M2, delta_b)
    print(label)

    solver = EntropyModelAB()
    solver.load(label)

    # solver.read_entropy(label)
    solver.entropy_with_modelAB(current=False, reg=reg)
    solver.plot_entropy(label+'_phi')
