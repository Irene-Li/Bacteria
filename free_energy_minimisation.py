from minimise import *
from matplotlib import pyplot as plt
import time



def plot_free_energy(objs, A, B, Lmax):
    legends = ['square lattice', 'hexagonal lattice']
    Ls = np.linspace(1, Lmax, 1000)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)

    f_uni = objs[0].free_energy_for_uniform()
    f_laminar = objs[0].free_energy_for_laminar_range(Lmax, A, B)
    plt.axhline(y=f_uni, color='k', linestyle='--', label=r'uniform')
    plt.plot(Ls, f_laminar, label=r'lamellar')
    for (i, obj) in enumerate(objs):
        free_energy = obj.total_free_energy_for_range(Lmax, A, B)
        plt.plot(Ls, free_energy, label=legends[i])
    plt.ylim([1.01*np.amin(free_energy), 2*f_uni - 1.01*np.amin(free_energy)])
    plt.yticks([])
    plt.xticks(np.linspace(0, Lmax, 5))
    plt.ylabel(r'$V^{-1}\mathcal{F}$')
    plt.xlim([0, Lmax])
    plt.xlabel(r'$L$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('total_free_energy.pdf')
    plt.close()

def plot_minimise(obj, N):
    min_free_energy = np.load("minimums.npy")
    phit = obj.phit()
    step_size = -2*phit/(N-1)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=17)
    print(phit, min_free_energy.shape)


    # x = np.linspace(-phit-step_size*0.5, 2+phit+step_size*0.5, N+1)
    # y = np.linspace(-2-phit-step_size*0.5, phit+step_size*0.5, N+1)
    x = np.linspace(-phit, 2+phit, N)
    y = np.linspace(-2-phit, phit, N)
    i, j = np.unravel_index(np.argmin(min_free_energy), min_free_energy.shape)
    print(x[j], y[i])
    plt.contourf(x, y, min_free_energy, cmap='plasma')
    plt.colorbar(ticks=[0])
    plt.xlabel(r'$\phi_1$')
    plt.xticks([0.8, 1, 1.2])
    plt.yticks([-1.2, -1, -0.8])
    plt.ylabel(r'$\phi_2$')
    plt.tight_layout()
    plt.savefig('min_free_energy.pdf')
    plt.close()

def plot_phase_diagram(logm_min, logm_max, N):
    phases = np.load("phases.npy")
    logms = np.linspace(logm_min, logm_max, N)
    phits = np.linspace(-1, 0, N)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)

    plt.pcolor(logms, phits, phases, cmap="Blues")
    plt.xlabel(r'$\log(m)$')
    plt.ylabel(r'$\phi_\mathrm{t}$')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    a = 0.2
    b = 0.2
    k = 1
    c = 0.08
    m = np.sqrt(1e-4)
    Nterms = 20
    N = 41
    logm_min = -10
    logm_max = -3

    print(np.sqrt(2*k/a))

    sq_obj = FreeEnergySquare(a, b, k, c, m)
    tri_obj = FreeEnergyTriangle(a, b, k, c, m)
    sq_obj.initialise(Nterms)
    tri_obj.initialise(Nterms)
    plot_free_energy([sq_obj, tri_obj], 0.855, -0.855, 400)
    # phit = tri_obj.minimise_over_AB(1000, N)
    # plot_minimise(tri_obj, N)

    # start_time = time.time()
    # tri_obj.phases(logm_min, logm_max, N)
    # plot_phase_diagram(logm_min, logm_max, N)
    # end_time = time.time()
    # print(end_time - start_time)
