import time
import numpy as np
from DetEvolution1D import *

X = 400
deltas = np.array([1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2])
amp = []
n = []
for delta in deltas:
    label = 'X_{}_delta_{}_amp_eq'.format(X, delta)
    solver = DetEvolution1D()
    solver.load(label)
    solver.rescale_to_standard()
    # solver.plot_steady_state(label)
    amp.append(np.max(solver.phi[-2]))
    n.append(solver.count())

alpha = solver.a
k = solver.k
X = solver.X
# phi_shift = 100
# u_tilde = alpha**2/(4*k*(1+deltas))
# delta2 = alpha/np.sqrt(u_tilde*k)
# delta3 = 1/phi_shift
qc = np.sqrt(alpha/(2*k))
# xi = (k/u_tilde)**(1/4)
# lamd = qc*xi
# g = 3*lamd**2*delta2 - 2*delta3**2/(9*lamd**4)
x = np.log(deltas)
y = np.log(amp)

poly = np.poly1d(np.polyfit(np.log(deltas), np.log(amp), 1))

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=17)
plt.plot(x, y, 'x', label=r'simulations')
plt.plot(x, poly(x), '--', label=r'best fit, gradient={:.2f}'.format(poly.c[0]))
plt.ylabel(r'$\log(A)$')
plt.xlabel(r'$\log(\Delta)$')
plt.legend()
plt.tight_layout()

# plt.plot(np.log(deltas), np.log(n), 'o')
plt.savefig('amp.pdf')
plt.close()

# plt.plot(x, X/n, 'x', label=r'simulations')
# plt.axhline(y=2*np.pi/qc, color='k', label=r'$2\pi/q_\mathrm{c}$')
# plt.ylabel(r'Wavelength')
# plt.xlabel(r'$\log(\Delta)$')
# plt.legend()
# plt.tight_layout()
# plt.savefig('wavelength_ampeq.pdf')
# plt.close()
#
# X = 300
# deltas = [0.04, 0.1]
#
#
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=20)
# for delta in deltas:
#     label = 'X_{}_delta_{}_amp_eq'.format(X, delta)
#     solver = DetEvolution1D()
#     solver.load(label)
#     # solver.rescale_to_standard()
#     phi = solver.phi[-2]
#     plt.plot(phi, label='$\Delta={}$'.format(delta))
#
#
# plt.xticks([])
# plt.yticks([-1, 0, 1],
#             [r'$\phi_\mathrm{B}$', r'$\phi_\mathrm{t}$', r'$-\phi_\mathrm{B}$'])
# plt.ylim([-1.1, 1.1])
# plt.xlabel(r'$x$')
# plt.legend()
# plt.tight_layout()
# plt.savefig('amp_eq.pdf')
# plt.close()
