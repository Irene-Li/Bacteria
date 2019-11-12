import numpy as np
from scipy.special import jv
from matplotlib import pyplot as plt

a = 1
b = 1
k = 1

m = 0.01
c = 0.1
Nterms = 20

class FreeEnergy:

    def __init__(self, a, b, k, c, m):
        self.a = a
        self.b = b
        self.k = k
        self.c = c
        self.m = m
        self.sigma = np.sqrt(self.a*self.k/18)

    def initialise(self, Nterms):
        pass

    def free_energy_nl(self, L, A, B, ratio):
        chi = self.m*L*self.lattice_scale_factor/(2*np.pi)
        nx = np.arange(1, self.Nterms)
        ny = np.arange(0, self.Nterms)
        nx, ny = np.meshgrid(nx, ny)
        n = self._distance(nx, ny)
        Ek = self._Ek(n, ratio, chi)
        Ek[n>Nterms*self.lattice_scale_factor]=0
        sum = np.sum(Ek)
        term1 = self.lattice_sym*(chi*ratio)**2*sum
        area_ratio = np.pi/self.lattice_scale_factor*ratio**2
        term2 = (A*area_ratio+B*(1-area_ratio))**2
        return (term1+term2)/2

    def free_energy_local(self, L, A, B, ratio):
        area_ratio = np.pi/self.lattice_scale_factor*ratio**2
        term1 = self._f(A)*area_ratio + self._f(B)*(1-area_ratio)
        term1 += self.sigma*(A-B)**2*2*np.pi/self.lattice_scale_factor*ratio/L
        return term1

    def total_free_energy_for_range(self, Lmax, deltaL, A, B):
        Ls = np.arange(1, Lmax, deltaL)
        free_energy = np.empty_like(Ls, dtype='float64')
        r = self._ratio(A, B)
        for (i, L) in enumerate(Ls):
            free_energy[i] = self.free_energy_local(L, A, B, r)+self.free_energy_nl(L, A, B, r)
        return free_energy

    def min_free_energy(self, Lmax, A, B):
        r = self._ratio(A, B)
        minimum = 0
        for L in np.arange(1, Lmax, 1):
            f = self.free_energy_local(L, A, B, r)
            f+= self.free_energy_nl(L, A, B, r)
            minimum = min(f, minimum)
        return minimum

    def _f(self, x):
        return self.c*x - self.a*x**2/2 + self.b*x**4/4

    def _Ek(self, n, ratio, chi):
        term1 = jv(1, ratio*n*2*np.pi/self.lattice_scale_factor)/n
        term2 = chi**2 + n**2
        return term1**2/term2

    def _ratio(self, A, B):
        pass #to implement in subclasses

    def _distance(self, nx, ny):
        pass #to impletement in subclass



class FreeEnergySquare(FreeEnergy):

    def initialise(self, Nterms):
        self.Nterms = Nterms
        self.lattice_scale_factor = 1
        self.lattice_sym = 4

    def _ratio(self, A, B):
        return np.sqrt((- self.c - B**3)/(A**3 - B**3)/np.pi)

    def _distance(self, nx, ny):
        return np.sqrt(nx**2+ny**2)


class FreeEnergyTriangle(FreeEnergy):

    def initialise(self, Nterms):
        self.Nterms = Nterms
        self.lattice_scale_factor = np.sqrt(3)/2
        self.lattice_sym = 6

    def _ratio(self, A, B):
        return np.sqrt(np.sqrt(3)*(-self.c - B**3)/(2*np.pi*(A**3-B**3)))

    def _distance(self, nx, ny):
        return np.sqrt(nx**2+ny**2+nx*ny)


def plot_free_energy(objs, A, B, Lmax, deltaL):
    legends = ['square lattice', 'triangular lattice']
    Ls = np.arange(1, Lmax, deltaL)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)

    for (i, obj) in enumerate(objs):
        free_energy = obj.total_free_energy_for_range(Lmax, deltaL, A, B)
        plt.plot(Ls, free_energy, label=legends[i])

    plt.legend()
    plt.ylim([-0.25, 0])
    plt.ylabel(r'$V^{-1}\mathcal{F}$')
    plt.xlim([0, Lmax])
    plt.xlabel(r'$L$')
    plt.tight_layout()
    plt.savefig('total_free_energy.pdf')
    plt.close()

def minimise(obj, Lmax):
    As = np.arange(0.5, 2.1, 0.1)
    Bs = np.arange(-2, -0.4, 0.1)
    min_free_energy = np.empty((len(As), len(Bs)), dtype='float64')
    print(min_free_energy.shape)
    for (i, A) in enumerate(As):
        for (j, B) in enumerate(Bs):
            min_free_energy[j, i] = obj.min_free_energy(Lmax, A, B)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)

    x = np.arange(0.45, 2.15, 0.1)
    y = np.arange(-2.05, -0.35, 0.1)
    plt.pcolor(x, y, min_free_energy, edgecolors='face', cmap='plasma')
    plt.colorbar(ticks=[0, -0.1, -0.2])
    plt.xlabel(r'$\phi_1$')
    plt.xticks([0.5, 1, 1.5, 2])
    plt.yticks([-2, -1.5, -1, -0.5])
    plt.ylabel(r'$\phi_2$')
    plt.tight_layout()
    plt.savefig('min_free_energy.pdf')
    plt.close()



if __name__ == '__main__':
    a = 1
    b = 1
    k = 1
    c = 0.1
    m = 0.01
    Nterms = 20

    sq_obj = FreeEnergySquare(a, b, k, c, m)
    tri_obj = FreeEnergyTriangle(a, b, k, c, m)
    sq_obj.initialise(Nterms)
    tri_obj.initialise(Nterms)
    plot_free_energy([sq_obj, tri_obj], 1, -1, 1000, 1)
    # minimise(tri_obj, 1000)
