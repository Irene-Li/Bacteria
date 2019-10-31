import numpy as np
from matplotlib import pyplot as plt
from scipy.special import iv, kn
from scipy.optimize import root_scalar
from StoEvolution2D import *

class Droplet():

    def __init__(self, A, a=0.2, k=1, phi_shift=10, phi_target=-0.7, u=1e-5):
        self.phi_shift = phi_shift
        self.phi_target = phi_target
        self.u = u
        self.gamma = np.sqrt(8*k*a/9)/(4*a)
        self.D = 2*a
        self.A = A

    def calculate_params(self):
        gradient_dense = - self.u*(2 + self.phi_shift - self.phi_target)
        self.gradient_dilute = - self.u*(-2+self.phi_shift - self.phi_target)
        f_dense = - self.u*(1+self.phi_shift)*(1-self.phi_target)
        f_dilute = - self.u*(-1+self.phi_shift)*(-1-self.phi_target)
        self.c_dense = - f_dense/gradient_dense
        self.c_dilute = - f_dilute/self.gradient_dilute

        self.k_dense = np.sqrt(-gradient_dense/self.D)
        self.k_dilute = np.sqrt(-self.gradient_dilute/self.D)

    def set_boundary_conditions(self, keyword):
        if keyword == 'pbc':
            self.omega_minus = self.omega_minus_pbc
            self.omega_plus = self.omega_plus_pbc
            self._droplet_frac = self._droplet_frac_pbc
        elif keyword == 'finite':
            self.omega_minus = self.omega_minus_finite
            self.omega_plus = self.omega_plus_finite
            self._droplet_frac = self._droplet_frac_finite
        else:
            print("must have either pbc or finite as boundary condition")
    def plot_r_dot_single(self, us, phi_ts):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)
        for u in us:
            for phi_t in phi_ts:
                self.u = u
                self.phi_target = phi_t
                self.calculate_params()
                Rmin = self.gamma/self.c_dilute*0.1
                Rmax = 10/self.k_dilute
                R = np.arange(Rmin, Rmax, 0.01)
                r_dot = self.R_dot_single_droplet(R)
                plt.plot(R, r_dot, label=r"$-uM_\mathrm{{A}}\phi_\mathrm{{a}}={},\phi_\mathrm{{t}}={}$".format(u*self.phi_shift, phi_t))
        plt.axhline(y=0, c='k')
        plt.ylim([-0.003, 0.004])
        plt.yticks([0], [r'$0$'])
        Rmax = 1.2/self.k_dilute
        plt.xlim([0, Rmax])
        plt.xticks(np.arange(0, Rmax, 10))
        plt.xlabel(r'$R$')
        plt.ylabel(r'$\partial_t R$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('r_dot_single_droplet.pdf')
        plt.close()


    def plot_r_dot_mult(self, n_list):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=15)
        plt.axhline(y=0, c='k')
        ymax = 0
        ymin = -np.inf
        for N in n_list:
            (new_ymax, new_ymin) = self._plot_multiple_droplet(N)
            ymax = max(ymax, new_ymax)
            ymin = max(ymin, new_ymin)
        plt.legend()
        plt.ylim([ymin, 1.1*ymax])
        plt.yticks([0], [r'$0$'])
        Rmax = 0.5/self.k_dilute
        plt.xlim([0, Rmax])
        plt.xticks(np.arange(0, Rmax, 5))
        plt.xlabel(r'$R$')
        plt.tight_layout()
        plt.savefig('droplet.pdf')
        plt.close()

    def plot_n_against_r(self, labels):
        # extract n against r for simulations
        length = len(labels)
        R_sim= np.empty(length)
        error_sim = np.empty(length)
        N_sim = np.empty(length)
        for (i, label) in enumerate(labels):
            R_sim[i], error_sim[i], N_sim[i] = self._obtain_pattern_length_skimage(label)

        # extract theoretical predictions for similar range of N
        N_th = np.arange(min(N_sim)-1, max(N_sim)+1)
        R_th = np.empty_like(N_th, dtype='float64')
        Rmax = 100/self.k_dilute
        for (i, N) in enumerate(N_th):
            Rmin = self._find_root_of_omega_minus(N)
            r_dot_min = self.R_dot_mult_droplets(Rmin, N)
            r_dot_max = self.R_dot_mult_droplets(Rmax, N)
            assert r_dot_max < 0 and r_dot_min > 0
            R_th[i] = self._find_root_of_r_dot(N, Rmin, Rmax)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=15)
        plt.errorbar(N_sim, R_sim, yerr=error_sim, fmt='x', label='simulations')
        plt.plot(N_th, R_th, '--', label='theory')
        plt.xticks(N_th[::2])
        plt.legend()
        plt.xlabel(r'Number of droplets')
        plt.ylabel(r'$R_\mathrm{s}$')
        plt.tight_layout()
        plt.savefig('n_against_r.pdf')
        plt.close()

    def plot_epsilon_against_n(self, epsilons, labels):
        length = len(labels)
        N= np.empty(length)
        for (i, label) in enumerate(labels):
            _, _, N[i] = self._obtain_pattern_length_skimage(label)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=15)
        x = 1/np.asarray(epsilons)
        y = np.log(N)
        poly = np.poly1d(np.polyfit(x, y, 1))
        plt.plot(x, y, 'x', label='simulations')
        plt.plot(x, poly(x), '--', label=r'straight line fit')
        plt.xlabel(r'$\epsilon^{-1}$')
        plt.ylabel(r'$\log(N)$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('epsilon_against_n.pdf')
        plt.close()

    def plot_stable_radius(self):
        logus = np.arange(-5.5, -3, 0.01)
        us = np.power(10, logus)
        rs = np.empty_like(us)
        ns = np.empty_like(us)
        N = int(A)
        for (i, u) in enumerate(us):
            self.u = u
            self.calculate_params()
            r, n = self._find_stable_radius(N/2, 0, 1, 1, N)
            rs[i] = r
            ns[i] = n
        #
        # rs[rs==0]=1
        # ns[ns==0]=1
        lambdas = np.pi*rs*rs*ns/self.A

        plt.plot(logus, ns, '+')
        plt.savefig('num_of_droplets.pdf')
        plt.close()

        plt.plot(logus, lambdas, '+')
        plt.savefig('lambdas.pdf')
        plt.close()

        plt.plot(logus, rs, '+')
        plt.savefig('stable_radius.pdf')
        plt.close()

    def omega_plus_finite(self, R, N):
        droplet_frac = self._droplet_frac_finite(N, R)
        single_droplet_frac = droplet_frac/(N-1)
        J = self._J_dilute_multiple_droplets(R, droplet_frac)
        k1_k0 = self._k1_k0(R)
        a = self._dJidJj(R, single_droplet_frac, k1_k0)
        b = self._dJidRi_finite(R, J, N, droplet_frac, k1_k0)
        c = self._dJidRj(R, J, single_droplet_frac, k1_k0)
        g = self._dJdR_dense(R)
        return (g - (b-c)/(1-a))/2

    def omega_plus_pbc(self, R, N):
        droplet_frac = self._droplet_frac_pbc(N, R)
        single_droplet_frac = droplet_frac/N
        J = self._J_dilute_multiple_droplets(R, droplet_frac)
        k1_k0 = self._k1_k0(R)
        b = self._dJidRi_pbc(R, J, N, droplet_frac, k1_k0)
        c = self._dJidRj(R, J, single_droplet_frac, k1_k0)
        g = self._dJdR_dense(R)
        return (g - (b-c))/2

    def omega_minus_finite(self, R, N):
        droplet_frac = self._droplet_frac_finite(N, R)
        single_droplet_frac = droplet_frac/(N-1)
        J = self._J_dilute_multiple_droplets(R, droplet_frac)
        k1_k0 = self._k1_k0(R)
        a = self._dJidJj(R, single_droplet_frac, k1_k0)
        b = self._dJidRi_finite(R, J, N, droplet_frac, k1_k0)
        c = self._dJidRj(R, J, single_droplet_frac, k1_k0)
        g = self._dJdR_dense(R)
        return (g - (b+(N-1)*c)/(1+(N-1)*a))/2

    def omega_minus_pbc(self, R, N):
        droplet_frac = self._droplet_frac_pbc(N, R)
        single_droplet_frac = droplet_frac/N
        J = self._J_dilute_multiple_droplets(R, droplet_frac)
        k1_k0 = self._k1_k0(R)
        a = self._dJidJj(R, single_droplet_frac, k1_k0)
        b = self._dJidRi_pbc(R, J, N, droplet_frac, k1_k0)
        c = self._dJidRj(R, J, single_droplet_frac, k1_k0)
        g = self._dJdR_dense(R)
        return (g - (b+(N-1)*c)/(1+N*a))/2

    def R_dot_single_droplet(self, R):
        J_dense = self._J_dense(R)
        J_dilute = self._J_dilute_single_droplet(R)
        return (J_dense - J_dilute)/2

    def R_dot_mult_droplets(self, R, N):
        droplet_frac = self._droplet_frac(N, R)
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

    def _find_root_of_omega_minus(self, N):
        omega_minus = lambda r: self.omega_minus(r, N)
        min = self.gamma/self.c_dilute*0.1
        max = 1/self.k_dilute
        if (omega_minus(max)>0):
            print(omega_minus(max))
        sol = root_scalar(omega_minus, bracket=[min, max], xtol=0.01, method='brentq')
        return sol.root

    def _find_root_of_r_dot(self, N, Rmin, Rmax):
        growth_rate = lambda r : self.R_dot_mult_droplets(r, N)
        sol = root_scalar(growth_rate, bracket=[Rmin, Rmax], xtol=0.01, method='brentq')
        return sol.root

    def _find_stable_radius(self, N, R, tol, Nmin, Nmax):
        Rmin = self._find_root_of_omega_minus(N)
        Rmax = 100/self.k_dilute
        r_dot_min = self.R_dot_mult_droplets(Rmin, N)
        r_dot_max = self.R_dot_mult_droplets(Rmax, N)
        assert r_dot_max < 0
        if r_dot_min < 0:
            Nmax = N
            new_N = int((N+Nmin)/2)
            new_R = 0
            if Nmax-new_N <= tol:
                if Nmin <= 1:
                    return (R, N)
                else:
                    Rmin = self._find_root_of_omega_minus(Nmin)
                    R = self._find_root_of_r_dot(Nmin, Rmin, Rmax)
                    return (R, Nmin)
        else:
            new_R = self._find_root_of_r_dot(N, Rmin, Rmax)
            omega_plus = self.omega_plus(new_R, N)
            if omega_plus > 0:
                Nmax = N
                new_N = int((N+Nmin)/2)
            else:
                Nmin = N
                new_N = int((N+Nmax)/2)
                if Nmax-new_N <= tol:
                    # print("final result: R = {}, N = {}, Nmin = {}, Nmax = {}".format(R, N, Nmin, Nmax))
                    return (new_R, new_N)
        # print("R = {}, N = {}, Nmin = {}, Nmax = {}".format(R, N, Nmin, Nmax))
        return self._find_stable_radius(new_N, new_R, tol, Nmin, Nmax)

    def _plot_multiple_droplet(self, N):
        Rmin = self.gamma/self.c_dilute*0.8
        Rmax = 0.5/self.k_dilute
        R = np.arange(Rmin, Rmax, 0.01)
        R_dot = self.R_dot_mult_droplets(R, N)
        omega_plus = 1.5*self.omega_plus(R, N)
        plt.plot(R, R_dot, label=r'$\dot{{R}}$ for $N={}$'.format(N))
        plt.plot(R, omega_plus, label=r'$\omega_+$ for $N = {}$'.format(N))
        return np.max(R_dot), max(R_dot[0], R_dot[-1])

    def _obtain_pattern_length_skimage(self, label):
        solver = StoEvolution2D()
        solver.load(label)
        phi = solver.phi[-2]

        # use skimage to find all the circles
        from skimage.transform import hough_circle, hough_circle_peaks
        from skimage.feature import canny
        edges = canny(phi, sigma=5)
        try_radii = np.arange(5, int(phi.shape[0]/5))
        hough_space = hough_circle(edges, try_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_space, try_radii,
                                                    threshold=0.4)

        # filter out circles that are too close by
        indices = self._filter_circles(cx, cy, 20, phi.shape[0])
        radii = radii[indices]
        N = radii.size
        radii = np.delete(radii, np.argmin(radii)) # delete the smallest circle
        return np.mean(radii), np.std(radii), N

    def _filter_circles(self, cx, cy, min_distance, side_length):
        length = cx.size
        indices = np.ones(length, dtype=bool)
        min_sq = min_distance**2
        for i in reversed(range(length)):
            x = cx[i]
            y = cy[i]
            for j in range(0, i):
                x2 = cx[j]
                y2 = cy[j]
                dx = min(np.abs(x-x2), side_length-np.abs(x-x2))
                dy = min(np.abs(y-y2), side_length-np.abs(y-y2))
                indices[i] = (dx**2+dy**2 > min_sq) & indices[i]
        return indices

    def _plot_circles(self, cx, cy, radii, side_length):
        from skimage.draw import circle_perimeter
        image = np.zeros((side_length, side_length))
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius,
                                            shape=image.shape)
            image[circy, circx] = 1
        plt.imshow(image)
        plt.show()


    def _J_dense(self, R):
        z = self.k_dense*R
        J_dense = - self.D*self.k_dense*(self.gamma/R - self.c_dense)*iv(1, z)/iv(0, z)
        return J_dense

    def _J_dilute_single_droplet(self, R):
        factor = self._k1_k0(R)
        J_dilute = self.D*self.k_dilute*(self.gamma/R - self.c_dilute)*factor
        return J_dilute

    def _droplet_frac_finite(self, N, R):
        v1 = np.pi*R*R
        return v1*(N-1)/(self.A-v1)

    def _droplet_frac_pbc(self, N, R):
        v1 = np.pi*R*R
        return v1*N/self.A

    def _k1_k0(self, R):
        z = self.k_dilute*R
        return kn(1, z)/kn(0, z)

    def _J_dilute_multiple_droplets(self, R, droplet_frac):
        factor = self._k1_k0(R)
        term1 = (1-droplet_frac)*self.c_dilute - self.gamma/R
        term2 = (2*droplet_frac*self.D*self.k_dilute/(self.gradient_dilute*R))*factor
        J_dilute = - self.D*self.k_dilute*factor*term1/(1-term2)
        return J_dilute

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

    def _dJidRi_finite(self, R, J, N, droplet_frac, k1_k0):
        z = self.k_dilute*R
        term1 = 2*J/(self.gradient_dilute*R)
        term2 = -self.D*self.k_dilute**2*(1-k1_k0/z+k1_k0**2)
        term2 *= (1-droplet_frac)*self.c_dilute-self.gamma/R-droplet_frac*term1
        term3 = -self.D*self.k_dilute*k1_k0
        term3 *= self.gamma/(R*R)-2*droplet_frac**2/((N-1)*R)*(self.c_dilute+term1)
        return term2+term3

    def _dJidRi_pbc(self, R, J, N, droplet_frac, k1_k0):
        z = self.k_dilute*R
        term1 = J/(self.gradient_dilute*R)
        term2 = -self.D*self.k_dilute**2*(1-k1_k0/z+k1_k0**2)
        term2 *= (1-droplet_frac)*self.c_dilute-self.gamma/R-droplet_frac*term1*2
        term3 = -self.D*self.k_dilute*k1_k0
        term3 *= self.gamma/(R*R)-2*droplet_frac/(N*R)*(self.c_dilute+term1)
        return term2+term3


    def _dJidRj(self, R, J, single_droplet_frac, k1_k0):
        result = 2*single_droplet_frac*self.D*self.k_dilute/R
        result *= k1_k0*(self.c_dilute + J/(self.gradient_dilute*R))
        return result

    def _dJidJj(self, R, single_droplet_frac, k1_k0):
        return -2*single_droplet_frac*self.D*self.k_dilute/(self.gradient_dilute*R)*k1_k0


if __name__ == '__main__':
    phi_targets = [-0.7, -0.8]
    us = [1e-4, 5e-5]
    A = 256**2
    solver = Droplet(A)
    solver.set_boundary_conditions('pbc')
    solver.plot_r_dot_single(us, phi_targets)
    # solver.calculate_params()


    # epsilons = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12]
    # orders = ['_3', '_4', '_2', '_2', '', '', '', '']
    # labels = ['phi_t_{}_u_{}_epsilon_{}{}'.format(phi_target, u, epsilon, order)
    #             for (epsilon, order) in zip(epsilons, orders)]
    # solver.plot_n_against_r(labels)
    # solver.plot_epsilon_against_n(epsilons, labels)
