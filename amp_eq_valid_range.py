import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

a = 0.2
k = 1
phi_s = 10

phi_t = np.arange(-1, 0, 0.01)
u = np.exp(np.arange(-10, -1, 0.1))

def delta(u, phi_t):
    alpha_tilde = 1 - 3*phi_t**2
    u_tilde = u*(phi_s+phi_t)
    delta = alpha_tilde**2/(4*k*u_tilde) - 1
    delta[alpha_tilde<0]=-1
    return delta

def g(u, phi_t):
    alpha_tilde = 1 - 3*phi_t**2
    u_tilde = u*(phi_s+phi_t)
    qc = np.sqrt(alpha_tilde/(2*k))
    xi = (k/u_tilde)**(1/4)
    l = qc*xi
    delta2 = a/(u_tilde*xi**2)
    delta3 = u/u_tilde
    delta1 = 3*phi_t*delta2
    g = 27*l**6*delta2 - (2*delta1*l**2+2*delta3)*(4*delta1*l**2+delta3)
    return g

phi_t = np.linspace(-0.7, 0, 100)
logu = np.linspace(-10, -3, 100)
x,y=np.meshgrid(logu, phi_t)
delta_amp_eq = delta(np.exp(x), y)
g_amp_eq = g(np.exp(x), y)
shade = np.copy(delta_amp_eq)
shade[g_amp_eq<0.01]=-0.1
# plt.pcolor(x, y, np.log(delta(np.exp(x), y)), edgecolors='face', cmap='Blues', alpha=1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)

plt.contourf(x, y, delta(np.exp(x), y), [-10 , 0, 0.5, 5000], colors=['lightslategrey', 'lavender', 'lightsteelblue'])
plt.contourf(x, y, shade, [-10, 0.01, 0.49, 5000], colors=['white', 'red', 'white'], alpha=0.2)
plt.contour(x[:, :52], y[:, :52], g_amp_eq[:, :52], [0], linewidths=[3])
plt.xlabel(r'$\log(u M_\mathrm{A})$')
plt.ylabel(r'$\phi_\mathrm{t}$')
plt.text(-8, -0.2, r'$\Delta>0.5$',  {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
plt.text(-4, -0.6, r'$\Delta<0$',  {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
plt.tight_layout()
plt.savefig('amp_eq_valid_range.pdf', dpi=1200)
plt.close()
