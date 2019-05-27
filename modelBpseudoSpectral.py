import numpy as np
import sys, time
import scipy as sp
from scipy.io import savemat, loadmat
fft2  = np.fft.fft2
ifft2 = np.fft.ifft2
randn = np.random.randn
from matplotlib import pyplot as plt


#parameters of the model B: Δ(a*u + b*u*u*u + kΔu
try:
    Ng, Teff = int(sys.argv[1]), float(sys.argv[2])
except:
    Ng, Teff = 128, 0.2
    print( 'No input given. Taking the default values!')


class activeModels():
    '''Class to solve active models'''
    def __init__(self, Nt, dt, dd, rhs):
        self.Nt = Nt
        self.dt = dt
        self.dd = dd

        self.rhs = rhs
        self.XX  = np.zeros((int(self.dd+1), Ng*Ng))

    def integrate(self, u):
        '''  simulates the equation and plots it at different instants '''
        ii=0
        t1 = time.perf_counter()

        for i in range(self.Nt):
            #if time.perf_counter() - t1 > 42000:
            #    break
            u = u + self.dt*self.rhs(u)

            if i%(int(self.Nt/self.dd))==0:
                self.XX[ii,:] = (np.real(np.fft.ifft2(u))).flatten()
                ii += 1


# now set-up the simulation
a, b, k    = -0.25, 0.25, 1
rate, phi_s = 1e-5, 100
Nt, dt, dd = 5000001, .005, 1000
phi0, nfac = -0.4, np.sqrt(2*Teff/dt)
nfac2 = np.sqrt(2*Teff*rate*phi_s/dt)
print(nfac)

# Fourier grid.
Nx, Ny = Ng, Ng
kx = (2*np.pi/Nx) * np.concatenate((np.arange(0, Nx/2+1,1),np.arange(-Nx/2+1, 0, 1)))
ky = (2*np.pi/Ny) * np.concatenate((np.arange(0, Nx/2+1,1),np.arange(-Nx/2+1, 0, 1)))
kx, ky = np.meshgrid(kx, ky)
ksq = kx*kx + ky*ky

# dealiasing
kk1 = kx
kmax = np.max(np.abs(kk1))
filtr = np.ones_like(kk1)
filtr2 = np.ones_like(kk1)
filtr[np.where(np.abs(kk1)>kmax*2./3)] = 0.
filtr2[np.where(np.abs(kk1)>kmax*1./2)] = 0.

kk1 = ky
kmax = np.max(np.abs(kk1))
filtr_1 = np.ones_like(kk1)
filtr_12 = np.ones_like(kk1)
filtr_1[np.where(np.abs(kk1)>kmax*2./3)] = 0.
filtr_12[np.where(np.abs(kk1)>kmax*1./2)] = 0.

dAl = filtr*filtr_1
dA2 = filtr2*filtr_12

def rhs(u):
    '''
    returns the right hand side of \dot{phi} in active model H
    \dot{phi} = Δ(a*u + b*u*u*u + kΔu + λ(∇u)^2)
    '''
    u_kx = 1j*kx*u;   u_x = ifft2(u_kx)
    u_ky = 1j*ky*u;   u_y = ifft2(u_ky)
    ukpp = -ksq*u ;   upp = ifft2(ukpp)

    uc = np.fft.ifft2(u)
    duO = -ksq*( a*u + b*dA2*fft2(uc*uc*uc) - k*ukpp)
    duO += -rate*(dAl*(uc*uc)+phi_s*uc)

    duN = nfac*(1j*kx*fft2(randn(Ng,Ng)) + 1j*ky*fft2(randn(Ng,Ng)))
    duN = nfac2*(fft2(randn(Ng,Ng)))
    return duO + duN


am = activeModels(Nt, dt, dd, rhs)
u = phi0 + 0*(1-2*np.random.random((Ng,Ng)))
#u = pymaft.utils.bubble(u, 15)


# run the simulation now!
t1 = time.perf_counter()
am.integrate(np.fft.fft2(u))

# save data
savemat('N%s__u0%2.2f_a%4.4f_DA.mat'%(Ng, phi0, a), {'X':am.XX, 'a':a, 'b':b, 'k':k, 'Ng':Ng, 'Nt':am.Nt, 'dt':dt, 'nfac':nfac, 'Tsim':time.perf_counter()-t1})
