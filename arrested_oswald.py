import numpy as np
from matplotlib import pyplot as plt
from scipy.special import iv, kn

phi_target = -0.9
phi_shift = 10
M1 = 1
alpha = 0.2
kappa = 1
u = 1e-6

sigma = np.sqrt(8*kappa*alpha/9)
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

R = np.arange(10, 0.8/l, 0.01)

J_plus = k*(gamma/R - A_dense)*iv(1, k*R)/iv(0, k*R)
J_plus_approx = (gamma/R - A_dense)*k*k*R/2
J_minus = - l*(gamma/R - A_dilute)*kn(1, l*R)/kn(0, l*R)
J_minus_approx = - (gamma/R - A_dilute)/(- np.log(l*R)*R)

def g(v):
    extra_term = gamma*(v*v-1)/R**2
    term1 = (gamma/R - A_dense)*(k - iv(1,k*R)/(R*iv(0,k*R)))
    term2 = (gamma/R - A_dilute)*(l - kn(1, l*R)/(R*kn(0,l*R)))
    term3 = (k*iv(v-1,k*R)/iv(v,k*R)-v/R)*(extra_term - J_plus)
    term4 = (l*kn(v-1,l*R)/kn(v,l*R)-v/R)*(extra_term - J_minus)
    return term1+term2+term3-term4

def h(v):
    return v/R * (-J_minus + J_plus)

plt.axhline(y=0, color='k')
plt.plot(R, J_minus-J_plus, label='u={}'.format(u))
# plt.plot(R, (J_minus_approx - J_plus_approx), '--', label='approx u={}'.format(u))
plt.plot(R, g(2), label='g')
plt.plot(R, h(2), label='h')

plt.legend()
plt.title('phi_t ={}, alpha = {}'.format(phi_target, alpha))
plt.xlabel('R')
plt.ylabel('growth rate')
plt.savefig('arrested_oswald.pdf')
