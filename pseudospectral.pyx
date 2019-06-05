import numpy as np
cimport numpy as np
DTYPE = np.complex128
DTYPE_t = np.complex128_t

def make_k_grid(int size):
  cdef np.ndarray[np.double64_t, ndim=1] kx, ky
  kx = np.fft.fftfreq(size)*2*np.pi
  ky = np.fft.fftfreq(size)*2*np.pi
  cdef np.ndarray[np.double64_t, ndim=2] kx_grid, ky_grid
  kx_grid, ky_grid = np.meshgrid(kx, ky)
  cdef np.ndarray[np.double64_t, ndim=2] ksq
  ksq = kx_grid*kx_grid + ky_grid*kygrid
  return kx_grid, ky_grid, ksq

def make_filters(np.ndarray[np.double64_t, ndim=2] kx_grid, np.ndarray[np.double64_t, ndim=2] ky_grid):
  cdef double kmax
  cdef np.ndarray[np.uint8_t, ndim=2] filtr, filtr2
  kmax = np.max(np.abs(kx_grid))
  filtr = np.ones_like(kx_grid)
  filtr2 = np.ones_like(ky_grid)
  filtr[np.where(np.abs(kx_grid)>kmax*2./3)] = 0.
  filtr2[np.where(np.abs(kk1)>kmax*1./2)] = 0.

  cdef np.ndarray[np.uint8_t, ndim=2] dealiasing_double, dealiasing_triple
  dealiasing_double = filtr*np.transpose(filtr)
  dealiasing_triple = filtr2*np.transpose(filtr2)
  return dealiasing_double, dealiasing_triple

def evolve(np.ndarray[DTYPE_t, ndim=2] init, double a, double k, double M1, double u, double phi_s, double phi_t, double dt, int n_itr, int batch_size, int size):
    cdef np.ndarray[np.double64_t, ndim=2] kx_grid, ky_grid, ksq
    kx_grid, ky_grid, ksq = make_k_grid(size)
    cdef np.ndarray[np.uint8_t, ndim=2] filtr, filtr2
    filtr, filtr2 = make_filters(kx_grid, ky_grid)
    cdef np.ndarray[np.double64_t, ndim=3] phi_evol
    cdef np.ndarray[DTYPE_t, ndim=2] phi

    phi = init
    n = 0
		for i in range(n_itr):
			if i % batch_size == 0:
				phi[n] = np.real(np.fft.ifft2(phi))
				n += 1
      cdef np.ndarray[DTYPE_t, ndim=2] phi_x, phi_cube, phi_sq
      phi_x = np.fft.ifft2(phi)
  		phi_cube = filtr2 * np.fft.fft2(phi_x**3)
  		phi_sq = filtr * np.fft.fft2(phi_x**2)

  		mu = -M1*ksq(a*(-phi+phi_cube) + k*ksq*phi)
  		birth_death = - self.u*(phi_sq+(self.phi_shift-self.phi_target)*phi)
  		birth_death[0, 0] += self.u*self.phi_shift*self.phi_target*self.size**2
  		dphidt = -M1*ksq(a*(-phi+phi_cube) + k*ksq*phi) + birth_death
