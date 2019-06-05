import numpy as np
from matplotlib import pyplot as plt
from scipy.special import iv, kn

phi_target = -0.8
phi_shift = 10
M1 = 1
alpha = 0.2
kappa = 1

plt.axhline(y=0, color='k')
for u in [1e-4, 5e-5, 1e-5]:
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
    delta_coefficient = sigma/(4*alpha)

    R = np.arange(2, 0.8/l, 0.01)

    J_plus = k*(delta_coefficient/R - A_dense)*iv(1, k*R)/iv(0, k*R)
    J_plus_approx = (delta_coefficient/R - A_dense)*k*k*R/2
    J_minus = l*(delta_coefficient/R - A_dilute)*kn(1, l*R)/kn(0, l*R)
    J_minus_approx = (delta_coefficient/R - A_dilute)/(- np.log(l*R)*R)
    plt.plot(R, - J_minus - J_plus, label='u={}'.format(u))
    plt.plot(R, (- J_minus_approx - J_plus_approx), '--', label='approx u={}'.format(u))
    # print(sigma/(4*alpha*A_dilute))
plt.legend()
plt.xlabel('R')
plt.ylabel('growth rate')
plt.savefig('arrested_oswald.pdf')
