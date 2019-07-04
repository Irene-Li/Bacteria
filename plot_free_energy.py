import numpy as np
import matplotlib.pyplot as plt

phi_s = 10
phi_t = -0.5
u = 0.09
left = -1.7
right = 1.7
phi = np.arange(left, right, 0.01)
def f_b(phi):
    return  - phi**2/2 + phi**4/4
def f_a(phi):
    return u*(1/3*phi**3 + 1/2*(phi_s-phi_t)*phi**2 - phi_s*phi_t*phi)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

fig,ax = plt.subplots(1)

ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.plot(phi, f_b(phi), label=r'Conservative')
ax.plot(phi, f_a(phi), label=r'Non-conservative')
ax.stem([-1, phi_t, 1], [f_b(-1), f_a(phi_t), f_b(1)], 'k--')
plt.xticks((-1, phi_t, 1), ('$-\phi_\mathrm{B}$', '$\phi_\mathrm{t}$','$\phi_\mathrm{B}$'))
plt.yticks([],[])
ax.set_ylabel(r'Free energy')
ax.set_xlim([left, right])
ax.legend()
plt.savefig('ab_f.pdf')
