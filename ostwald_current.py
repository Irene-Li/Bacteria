import numpy as np
from matplotlib import pyplot as plt
from scipy.special import iv, kn
from scipy.optimize import root_scalar

# Define the other parameters
phi_shift = 10
M1 = 1
alpha = 1
kappa = 1
sigma = np.sqrt(8*kappa*alpha/9)
u = 1e-5
phi_target = -0.7


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

def growth_rate(R):
    J_plus = - k*(gamma/R - A_dense)*iv(1, k*R)/iv(0, k*R)
    J_minus = l*(gamma/R - A_dilute)*kn(1, l*R)/kn(0, l*R)
    return J_plus - J_minus

min = gamma/A_dilute*1.5
max = 1/l
sol = root_scalar(growth_rate, bracket=[min, max], xtol=0.01, method='brentq')
Rc = sol.root


R_plus = np.arange(0, Rc, 0.1)
R_minus = np.arange(Rc, max, 0.1)
J_plus = - k*(gamma/Rc - A_dense)*iv(1, k*R_plus)/iv(0, k*Rc)
J_minus = l*(gamma/Rc - A_dilute)*kn(1, l*R_minus)/kn(0, l*Rc)
phi_plus = A_dense + (gamma/Rc - A_dense)*iv(0, k*R_plus)/iv(0, k*Rc)
phi_minus = A_dilute + (gamma/Rc - A_dilute)*kn(0, l*R_minus)/kn(0, l*Rc)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

plt.axhline(y=0, color='k')
plt.plot(R_plus, 100*J_plus, label=r'$J_+$')
plt.plot(R_minus, 100*J_minus, label=r'$J_-$')
plt.plot(R_plus, 1+phi_plus, label=r'$\phi_+$')
plt.plot(R_minus, -1+ phi_minus, label=r'$\phi_-$')
plt.xlabel(r'$R$')
plt.legend()
plt.tight_layout()
plt.show()
