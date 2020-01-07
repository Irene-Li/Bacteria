import numpy as np
from matplotlib import pyplot as plt
from scipy.special import iv, kn
from scipy.optimize import root_scalar
from scipy.ndimage.filters import gaussian_filter


# Define the other parameters
phi_shift = 10
M1 = 1
alpha = 0.2
kappa = 1
sigma = np.sqrt(8*kappa*alpha/9)

# Plot graph
phi_ts = np.arange(-0.99, -0.6, 0.002)
us = np.exp(np.arange(-15, -8, 0.02))
# phi_ts = np.arange(-0.99, 0, 0.001)
# us = np.exp(np.arange(-14.5, -7.5, 0.02))

g_v = np.empty((len(phi_ts), len(us)), dtype=np.float64)
radii = np.empty((len(phi_ts), len(us)), dtype=np.float64)

def spinodal(phi_t, u):
    alpha_tilde = alpha*(1-3*phi_t**2)
    delta = alpha_tilde - np.sqrt(4*kappa*u*phi_shift)
    return (delta > 0)

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
        max = 100/l
        if spinodal(phi_target, u):
            radii[i, j] = 0
            g_v[i, j] = 0
        elif (growth_rate(min)<0) and (growth_rate(max)<0):
            g_v[i, j] = 0
            radii[i, j] = 2
        elif (growth_rate(min)>0) and (growth_rate(max)<0):
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
            radii[i, j] = 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)
max = np.max(np.abs(g_v))
min = - max
x, y = np.meshgrid(np.log10(us*phi_shift), phi_ts)

plt.pcolor(x, y, g_v, vmin=min, vmax=max, edgecolors='face', cmap='seismic', alpha=1)
plt.colorbar(ticks=[0])
cs = plt.contour(x, y, g_v, [0])
plt.clabel(cs, fontsize=18, inline_spacing=8, fmt={0:r'$j_2 = 0.0$'})
plt.yticks(np.arange(-1, -0.6, 0.1))
plt.xlabel(r'$\log(-u M_\mathrm{A} \phi_\mathrm{a})$')
plt.ylabel(r'$\phi_\mathrm{t}$')
plt.title(r'$j_2(R_\mathrm{S})$')
plt.text(-3.3, -0.93, r'No stable radius',  {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
plt.tight_layout()
plt.savefig('stability.pdf')
plt.close()

# plt.contourf(x, y, radii, levels=[-0.5, 0.5, 1.5, 2.5], cmap=plt.cm.Blues)
# plt.xlabel(r'$\log(- u M_\mathrm{A} \phi_\mathrm{a})$')
# plt.ylabel(r'$\phi_\mathrm{t}$')
# plt.text(-2.9, -0.9, r'uniform',  {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
#           'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
# plt.text(-3.7, -0.3, r'spinodal decomposition',  {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
#           'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
# plt.text(-4.2, -0.71, r'nucleation', {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
#           'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
# plt.tight_layout()
# plt.savefig('spinodal_binodal.pdf')
# plt.close()
