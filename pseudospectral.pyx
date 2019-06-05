import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc.math cimport sqrt



@cython.wraparound(False)
@cython.boundscheck(False)
def make_k_grid(int size):
	cdef double [::1] kx
	kx = np.fft.fftfreq(size)*2*np.pi
	cdef double [:, ::1] kx_grid, ky_grid
	kx_grid, ky_grid = np.meshgrid(kx, kx)
	cdef Py_ssize_t i, j
	cdef double [:, ::1] ksq = np.empty((size, size))
	for i in range(size):
		for j in range(size):
			ksq[i, j] = kx_grid[i, j]**2 + ky_grid[i, j]**2
	return kx_grid, ky_grid, ksq

@cython.wraparound(False)
@cython.boundscheck(False)



@cython.wraparound(False)
@cython.boundscheck(False)
def evolve(np.complex128_t [:, :] init, double a, double k, double u, double phi_s, double phi_t, double epsilon, double dt, int nitr, int batch_size, int size):
	cdef double [:, ::1] kx_grid, ky_grid, ksq
	kx_grid, ky_grid, ksq = make_k_grid(size)
	cdef double [:, :, :] phi_evol
	cdef np.complex128_t [:, :] phi, phi_cube, phi_sq
	cdef np.ndarray[np.complex128_t, ndim=2] phi_x
	cdef np.complex128_t [:, :] dW
	cdef Py_ssize_t n, i, j, m
	cdef double M1, M2, kmax
	cdef np.complex128_t birth_death, noise, mu
	M1 = 1.0
	M2 = u*(phi_s + phi_t/2)
	kmax = np.pi

	phi = init
	phi_evol = np.empty((nitr, size, size), dtype=np.float64)
	n = 0
	for i in range(nitr):
		# if i % batch_size == 0:
		# 	phi_evol[n, :, ::1] = np.real(np.fft.ifft2(phi))
		# 	n += 1
		phi_x = np.fft.ifft2(phi)
		phi_cube = np.fft.fft2(phi_x**3)
		phi_sq = np.fft.fft2(phi_x**2)

		dW = np.fft.fft2(np.random.normal(size=(size, size)))

		for j in range(size):
			for m in range(size):
				if (kx_grid[j,m]>kmax/2) or (ky_grid[j,m]>kmax/2):
					phi_cube[j,m] = 0
					if (kx_grid[j,m]>kmax*2/3) or (ky_grid[j,m]>kmax*2/3):
						phi_sq[j,m] = 0
				birth_death = - u*(phi_sq[j, m]+(phi_s-phi_t)*phi[j,m])
				noise = sqrt(2*(M2+M1*ksq[j,m])*epsilon*dt)*dW[j,m]
				mu = a*(-phi[j,m]+phi_cube[j,m]) + k*ksq[j,m]*phi[j,m]
				phi[j,m] = dt*(-M1*ksq[j,m]*mu+birth_death)+noise + phi[j,m]
		phi[0,0] = u*phi_s*phi_t*size**2 + phi[0, 0]

	return phi_evol
