from minimise import *
from matplotlib import pyplot as plt
import time



def plot_free_energy(objs, A, B, Lmax):
    legends = ['square lattice', 'hexagonal lattice']
    Ls = np.linspace(1, Lmax, 1000)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)

    for (i, obj) in enumerate(objs):
        free_energy = obj.total_free_energy_for_range(Lmax, A, B)
        plt.plot(Ls, free_energy, label=legends[i])
    f_uni = objs[0].free_energy_for_uniform()
    f_laminar = objs[0].free_energy_for_laminar_range(Lmax, A, B)
    plt.axhline(y=f_uni, linestyle='--', label=r'uniform state')
    plt.plot(Ls, f_laminar, label=r'laminar')
    plt.legend()
    plt.ylim([1.1*min(free_energy), 2*f_uni - min(free_energy)])
    plt.ylabel(r'$V^{-1}\mathcal{F}$')
    plt.xlim([0, Lmax])
    plt.xlabel(r'$L$')
    plt.tight_layout()
    plt.show()
    # plt.savefig('total_free_energy.pdf')
    plt.close()

def plot_minimise(obj, Lmax):
    step_size = 0.05
    low = 0.6
    high = 1.8
    As = np.arange(low, high+step_size, step_size)
    Bs = np.arange(-high, -low+step_size, step_size)
    min_free_energy = np.empty((len(As), len(Bs)), dtype='float64')
    print(min_free_energy.shape)
    for (i, A) in enumerate(As):
        for (j, B) in enumerate(Bs):
            min_free_energy[j, i] = obj.min_free_energy(Lmax, A, B)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)

    x = np.arange(low-step_size*0.5, high+step_size*1.49, step_size)
    y = np.arange(-high-step_size*0.5, -low+step_size*1.49, step_size)
    print(x.shape, y.shape)
    plt.pcolor(x, y, min_free_energy, edgecolors='face', cmap='plasma')
    plt.colorbar(ticks=[0, -0.02, -0.04])
    plt.xlabel(r'$\phi_1$')
    plt.xticks([0.6, 1, 1.4, 1.8])
    plt.yticks([-1.8, -1.4, -1, -0.6])
    plt.ylabel(r'$\phi_2$')
    plt.tight_layout()
    plt.savefig('min_free_energy.pdf')
    plt.close()

def phase_diagram(obj, Lmax):
    logms = np.arange(-10, -2, 2)
    ms = np.exp(logms)
    phits = np.arange(-1, 0, 0.2)
    phases = np.empty((len(phits), len(ms)), dtype='int')
    for (i, m) in enumerate(ms):
        for (j, phit) in enumerate(phits):
            obj.set_m_phit(m, phit)
            print(m, phit)
            Lmax = 10/m
            lat_min, lam_min = obj.minimise(Lmax, phit)
            uniform = obj.free_energy_for_uniform()
            phases[j, i] = np.argmin([uniform, lat_min, lam_min])
    # plt.pcolor(logms, phits, phases)
    # plt.show()

if __name__ == '__main__':
    a = 0.2
    b = 0.2
    k = 1
    c = 0.0
    m = np.sqrt(1e-4)
    Nterms = 20

    print(np.sqrt(2*k/a))

    sq_obj = FreeEnergySquare(a, b, k, c, m)
    tri_obj = FreeEnergyTriangle(a, b, k, c, m)
    sq_obj.initialise(Nterms)
    tri_obj.initialise(Nterms)
    # plot_free_energy([sq_obj, tri_obj], 1, -1, 1000)

    start_time = time.time()
    phase_diagram(tri_obj, 800)
    end_time = time.time()
    print(end_time - start_time)
