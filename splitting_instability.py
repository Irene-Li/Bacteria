import numpy as np
from matplotlib import pyplot as plt
from scipy.special import iv, kn
from scipy.optimize import root_scalar


# Define the other parameters
phi_shift = 10
M1 = 1
alpha = 0.2
kappa = 1
sigma = np.sqrt(8*kappa*alpha/9)

# Plot graph
phi_ts = np.arange(-0.99, -0.6, 0.002)
us = np.exp(np.arange(-15, -7, 0.05))

g_v = np.empty((len(phi_ts), len(us)), dtype=np.float64)
radii = np.empty((len(phi_ts), len(us)), dtype=np.float64)

for (i, phi_target) in enumerate(phi_ts):
    for (j, u) in enumerate(us):
        # Calculate gamma, k, l, A_dilute, A_dense
        gradient_dense = - u*(2 + phi_shift - phi_target)
        gradient_dilute = - u*(-2+phi_shift - phi_target)
        f_dense = - u*(1+phi_shift)*(1-phi_target)
        f_dilute = - u*(-1+phi_shift)*(-1-phi_target)
        A_dense = - f_dense/gradient_dense
        A_dilute = - f_dilute/gradient_dilute

        D = 2*M1*alpha
        k = np.sqrt(-gradient_dense/D)
        l = np.sqrt(-gradient_dilute/D)
        gamma = sigma/(4*alpha)

        # Find the stable radius
        def growth_rate(R):
            J_plus = - k*(gamma/R - A_dense)*iv(1, k*R)/iv(0, k*R)
            J_minus = l*(gamma/R - A_dilute)*kn(1, l*R)/kn(0, l*R)
            return J_plus - J_minus

        min = gamma/A_dilute*1.1
        max = 1/l
        if growth_rate(min)<0:
            g_v[i, j] = 0
            radii[i, j] = 0
        else:
            sol = root_scalar(growth_rate, bracket=[min, max], xtol=0.01, method='brentq')
            v = 2
            R = sol.root
            b0_dense = (gamma/R - A_dense)/iv(0, k*R)
            b0_dilute = (gamma/R - A_dilute)/kn(0, l*R)
            extra_term = gamma*(v*v-1)/R**2
            term1 = b0_dense*k*k*(iv(0,k*R) - iv(1,k*R)/(k*R))
            term2 = -b0_dilute*l*l*(kn(0,l*R) - kn(1,l*R)/(l*R))
            term3 = (k*iv(v-1,k*R)/iv(v,k*R)-v/R)*(extra_term - b0_dense*k*iv(1, k*R))
            term4 = (l*kn(v-1,l*R)/kn(v,l*R)-v/R)*(extra_term + b0_dilute*l*kn(1, l*R))
            g_v[i,j] = -term1+term2-term3+term4
            radii[i, j] = R
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
max = np.max(np.abs(g_v))
min = - max
x, y = np.meshgrid(np.log10(us), phi_ts)
plt.pcolor(x, y, g_v, vmin=min, vmax=max, edgecolors='face', cmap='seismic', alpha=1)
plt.colorbar()
plt.xlabel(r'$\log(u)$')
plt.ylabel(r'$\phi_\mathrm{t}$')
plt.title(r'$g_l(\bar{R})$ for $l=2$')
plt.text(-4.3, -0.93, r'No stable radius',  {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
plt.tight_layout()
plt.savefig('stability.pdf')
plt.close()


cmap = plt.cm.plasma
cmap.set_under(color='white')

plt.pcolor(x, y, radii, edgecolors='face', vmin=0.0001, cmap=cmap, alpha=1)
plt.colorbar()
plt.xlabel(r'$\log(u)$')
plt.ylabel(r'$\phi_\mathrm{t}$')
plt.title(r'Critical radius')
plt.text(-4.3, -0.93, r'No stable radius',  {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
plt.tight_layout()
plt.savefig('radii.pdf')
plt.close()
