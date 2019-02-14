import numpy as np
import matplotlib.pyplot as plt
from Entropy import *

rates = [1e-7, 5e-7, 1e-6, 2.5e-6, 5e-6, 1e-5, 1.5e-5, 1.6e-5,
        1.7e-5, 1.8e-5, 1.9e-5, 2e-5, 2.1e-5, 2.2e-5, 2.3e-5, 2.5e-5]

l = len(rates)

us = np.zeros(l)
S = np.zeros(l)
epsilon = np.zeros(l)


for (i, u) in enumerate(rates):
    label = "X_200_u_{}".format(u)
    solver = EntropyProductionFourier()
    solver.load(label)
    solver.read_entropy(label)
    S[i] = np.sum(solver.entropy)
    us[i] = solver.u
    epsilon[i] = solver.a**2/(4*solver.k) - solver.u

plt.plot(us, S, 'x')
plt.show()

plt.plot(epsilon, S, 'x')
plt.show()
