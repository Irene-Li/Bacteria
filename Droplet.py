import numpy as np
from matplotlib import pyplot as plt
from scipy.special import iv, kn
from scipy.optimize import root_scalar

class Droplet():

    def __init__(self, a=0.2, k=1, phi_shift=10, phi_target=-0.7, u=1e-5):
        self.phi_shift = phi_shift
        self.phi_target = phi_target
        self.u = u
        self.gamma = np.sqrt(8*k*a/9)/(4*a)
        self.D = 2*a

    def calculate_params(self):
        gradient_dense = - self.u*(2 + self.phi_shift - self.phi_target)
        self.gradient_dilute = - self.u*(-2+self.phi_shift - self.phi_target)
        f_dense = - self.u*(1+self.phi_shift)*(1-self.phi_target)
        f_dilute = - self.u*(-1+self.phi_shift)*(-1-self.phi_target)
        self.c_dense = - f_dense/gradient_dense
        self.c_dilute = - f_dilute/self.gradient_dilute

        self.k_dense = np.sqrt(-gradient_dense/self.D)
        self.k_dilute = np.sqrt(-self.gradient_dilute/self.D)

    def _J_dense(self, R):
        z = self.k_dense*R
        J_dense = - self.D*self.k_dense*(self.gamma/R - self.c_dense)*iv(1, z)/iv(0, z)
        return J_dense

    def _J_dilute_single_droplet(self, R):
        factor = self._k1_k0(R)
        J_dilute = self.D*self.k_dilute*(self.gamma/R - self.c_dilute)*factor
        return J_dilute

    def _droplet_frac(self, N, R, A):
        v1 = np.pi*R*R
        return v1*(N-1)/(A-v1)

    def _k1_k0(self, R):
        z = self.k_dilute*R
        return kn(1, z)/kn(0, z)

    def _J_dilute_multiple_droplets(self, R, droplet_frac):
        factor = self._k1_k0(R)
        term1 = (1-droplet_frac)*self.c_dilute - self.gamma/R
        term2 = (2*droplet_frac*self.D*self.k_dilute/(self.gradient_dilute*R))*factor
        J_dilute = - self.D*self.k_dilute*factor*term1/(1-term2)
        return J_dilute

    def R_dot_single_droplet(self, R):
        J_dense = self._J_dense(R)
        J_dilute = self._J_dilute_single_droplet(R)
        return (J_dense - J_dilute)/2

    def R_dot_mult_droplets(self, R, N, A):
        droplet_frac = self._droplet_frac(R, N, A)
        J_dense = self._J_dense(R)
        J_dilute = self._J_dilute_multiple_droplets(R, droplet_frac)
        return (J_dense - J_dilute)/2

    def g_v(self, R, v):
        z_dense = self.k_dense*R
        z_dilute = self.k_dilute*R
        b0_dense = (self.gamma/R - self.c_dense)/iv(0, z_dense)
        b0_dilute = (self.gamma/R - self.c_dilute)/kn(0, z_dilute)

        extra_term = self.gamma*(v*v-1)/R**2

        term1 = b0_dense*(self.k_dense**2)*(iv(0,z_dense) - iv(1,z_dense)/z_dense)
        term2 = -b0_dilute*(self.k_dilute**2)*(kn(0,z_dilute) - kn(1,z_dilute)/z_dilute)
        term3 = (self.k_dense*iv(v-1,z_dense)/iv(v,z_dense)-v/R)*(extra_term - b0_dense*k*iv(1, z_dense))
        term4 = (self.k_dilute*kn(v-1,z_dilute)/kn(v,z_dilute)-v/R)*(extra_term + b0_dilute*l*kn(1, z_dilute))

        return -term1+term2-term3+term4

    def _dJdR_dense(self, R):
        z = self.k_dense*R
        i1 = iv(1, z)
        i0 = iv(0, z)
        term1 = self.c_dense - self.gamma/R
        term2 = 1 - i1/(self.k_dense*R*i0) - i1*i1/(i0*i0)
        result = self.k_dense*term1*term2 + (i1/i0)*self.gamma/R**2
        return result*self.k_dense

    def _dJdR_dilute(self, R):
        factor = self._k1_k0(R)
        term1 = self.c_dilute - self.gamma/R
        term2 = 1 - factor/(self.k_dilute*R) + factor*factor
        result = self.k_dilute*term1*term2 + factor*self.gamma/R**2
        return -result*self.k_dilute

    def _dJidRi(self, R, J, N, droplet_frac, k1_k0):
        z = self.k_dilute*R
        term1 = 2*J/(self.gradient_dilute*R)
        term2 = -self.D*self.k_dilute**2*(1-k1_k0/z+k1_k0**2)
        term2 *= (1-droplet_frac)*self.c_dilute-self.gamma/R-droplet_frac*term1
        term3 = -self.D*self.k_dilute*k1_k0
        term3 *= self.gamma/(R*R)-2*droplet_frac**2/((N-1)*R)*(self.c_dilute+term1)
        return term2+term3

    def _dJidRj(self, R, J, single_droplet_frac, k1_k0):
        result = 2*single_droplet_frac*self.D*self.k_dilute/R
        result *= k1_k0*(self.c_dilute + J/(self.gradient_dilute*R))
        return result

    def _dJidJj(self, R, single_droplet_frac, k1_k0):
        return -2*single_droplet_frac*self.D*self.k_dilute/(self.gradient_dilute*R)*k1_k0

    def omega_plus_small_lambda(self, R, A):
        v1 = np.pi*R*R
        single_droplet_frac = v1/(A-v1)
        factor = self._k1_k0(R)
        term1 = 1 - factor/(self.k_dilute*R) + factor**2
        term2 = self.c_dilute - self.gamma/R
        term3 = self.D*self.k_dilute/(self.gradient_dilute*R)
        term4 = self.c_dilute - term2*term3
        term5 = self.c_dilute/R * factor * (1 - term3)
        result = self.k_dilute*term1*term4 +term5
        result *= 2*single_droplet_frac*self.k_dilute*factor
        result += self._dJdR_dense(R) - self._dJdR_dilute(R)
        return result

    def omega_plus(self, R, N, A):
        droplet_frac = self._droplet_frac(N, R, A)
        single_droplet_frac = droplet_frac/(N-1)
        J = self._J_dilute_multiple_droplets(R, droplet_frac)
        k1_k0 = self._k1_k0(R)
        a = self._dJidJj(R, single_droplet_frac, k1_k0)
        b = self._dJidRi(R, J, N, droplet_frac, k1_k0)
        c = self._dJidRj(R, J, single_droplet_frac, k1_k0)
        g = self._dJdR_dense(R)
        return (g - (b-c)/(1-a))/2

    def omega_minus(self, R, N, A):
        droplet_frac = self._droplet_frac(N, R, A)
        single_droplet_frac = droplet_frac/(N-1)
        J = self._J_dilute_multiple_droplets(R, droplet_frac)
        k1_k0 = self._k1_k0(R)
        a = self._dJidJj(R, single_droplet_frac, k1_k0)
        b = self._dJidRi(R, J, N, droplet_frac, k1_k0)
        c = self._dJidRj(R, J, single_droplet_frac, k1_k0)
        g = self._dJdR_dense(R)
        return (g - (b+(N-1)*c)/(1+(N-1)*a))/2

    def _find_root_of_omega_minus(self, N, A):
        omega_minus = lambda r: self.omega_minus(r, N, A)
        min = self.gamma/self.c_dilute*0.1
        max = 1/self.k_dilute
        if (omega_minus(max)>0):
            print(omega_minus(max))
        sol = root_scalar(omega_minus, bracket=[min, max], xtol=0.01, method='brentq')
        return sol.root



    def plot_multiple_droplet(self, N, A):
        Rmin = self.gamma/self.c_dilute*0.8
        Rmax = 0.5/self.k_dilute
        R = np.arange(Rmin, Rmax, 0.1)
        R_dot = self.R_dot_mult_droplets(R, N, A)
        omega_plus = 0.05*self.omega_plus(R, N, A)
        plt.plot(R, R_dot, label='N={}, r_dot'.format(N))
        plt.plot(R, omega_plus, label='N = {}, omega'.format(N))

    def _find_stable_radius(self, N, A, tol, Nmin, Nmax):
        Rmin = self._find_root_of_omega_minus(N, A)
        Rmax = 100/self.k_dilute
        r_dot_min = self.R_dot_mult_droplets(Rmin, N, A)
        r_dot_max = self.R_dot_mult_droplets(Rmax, N, A)
        assert r_dot_max < 0
        if r_dot_min < 0:
            Nmax = N
            new_N = int((N+Nmin)/2)
            R = 0
        else:
            growth_rate = lambda r : self.R_dot_mult_droplets(r, N, A)
            sol = root_scalar(growth_rate, bracket=[Rmin, Rmax], xtol=0.01, method='brentq')
            R = sol.root
            omega_plus = self.omega_plus(R, N, A)
            if omega_plus > 0:
                Nmax = N
                new_N = int((N+Nmin)/2)
            else:
                Nmin = N
                new_N = int((N+Nmax)/2)
        if min(new_N-Nmin, Nmax-new_N) <= tol:
            # print("final result: R = {}, N = {}, Nmin = {}, Nmax = {}".format(R, N, Nmin, Nmax))
            return (R, N)
        else:
            return self._find_stable_radius(new_N, A, tol, Nmin, Nmax)

    def plot_stable_radius(self, A):
        us = np.exp(np.arange(-12, -6, 0.01))
        rs = np.empty_like(us)
        ns = np.empty_like(us)
        N = int(A/1e4)
        for (i, u) in enumerate(us):
            self.u = u
            self.calculate_params()
            r, n = self._find_stable_radius(int(N/2), A, 1, 1, N)
            rs[i] = r
            ns[i] = n

        rs[rs==0]=1
        ns[ns==0]=1

        plt.plot(np.log(us), np.log(ns), 'x')
        plt.show()

        plt.plot(np.log(us), np.log(rs), 'x')
        plt.show()

if __name__ == '__main__':
    phi_target = -0.7
    u = 5e-6
    A = 1e12
    solver = Droplet(phi_target=phi_target, u=u)
    # solver.calculate_params()
    # solver.plot_multiple_droplet(10, A)
    # solver.plot_multiple_droplet(20, A)
    # plt.axhline(y=0, c='k')
    # plt.legend()
    # plt.show()

    solver.plot_stable_radius(A)
