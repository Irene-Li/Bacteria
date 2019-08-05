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


# Plot single stability graph
phi_target = -0.7

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

plt.axhline(y=0, color='k')

for u in [1e-5]:

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

    def g_v(v, R):
        b0_dense = (gamma/R - A_dense)/iv(0, k*R)
        b0_dilute = (gamma/R - A_dilute)/kn(0, l*R)

        extra_term = gamma*(v*v-1)/R**2

        term1 = b0_dense*k*k*(iv(0,k*R) - iv(1,k*R)/(k*R))
        term2 = -b0_dilute*l*l*(kn(0,l*R) - kn(1,l*R)/(l*R))
        term3 = (k*iv(v-1,k*R)/iv(v,k*R)-v/R)*(extra_term - b0_dense*k*iv(1, k*R))
        term4 = (l*kn(v-1,l*R)/kn(v,l*R)-v/R)*(extra_term + b0_dilute*l*kn(1, l*R))

        return -term1+term2-term3+term4
    R = np.arange(1, 1/l, 0.1)
    plt.plot(R, growth_rate(R), label=r'$\dot{R}$')
    plt.plot(R, iv(1, k*R)/iv(0, k*R), label=r'$I_0/I_1$')
    plt.plot(R, kn(1, l*R)/kn(0, l*R), label=r'$K_0/K_1$')
    # plt.plot(R, g_v(2, R), label=r'$g_2(R)$'')

# plt.plot(R, g_v(2, R), label=r'$l=2$ mode')
# plt.ylim([-0.015, 0.02])
# plt.xlim([0, 60])
plt.xlabel(r'R')
plt.ylabel(r'growth rate')
plt.title(r'Growth rate')
plt.legend()
plt.tight_layout()
plt.show()
