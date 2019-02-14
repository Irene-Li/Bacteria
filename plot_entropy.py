import numpy as np
import matplotlib.pyplot as plt
from Entropy import *

rates = [1e-7, 5e-7, 1e-6, 2.5e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5,
        1.1e-5, 1.5e-5, 1.6e-5,
        1.7e-5, 1.8e-5, 1.9e-5, 2e-5, 2.1e-5, 2.2e-5, 2.3e-5, 2.5e-5]

l = len(rates)

us = np.zeros(l)
S = np.zeros(l)


for (i, u) in enumerate(rates):
    label = "X_200_u_{}".format(u)
    solver = EntropyProductionFourier()
    solver.load(label)
    solver.read_entropy(label)
    S[i] = np.sum(solver.entropy)
    us[i] = solver.u

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

plt.plot(us, S, 'k+', markersize=8, markeredgewidth=2)
plt.ylabel(r'$\dot{S}$')
plt.xlabel(r'$u$')
plt.title(r'Total entropy production against $u$')
plt.savefig("u_s.pdf")
