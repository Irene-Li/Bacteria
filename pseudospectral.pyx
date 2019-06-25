import numpy as np
import pyfftw
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fmin, M_PI

@cython.wraparound(False)
@cython.boundscheck(False)
def evolve(np.complex128_t [:, :] init, double a, double k, double u, double phi_s, double phi_t, double epsilon, double dt, int nitr, int batch_size, int size):
	cdef np.float64_t [:, :, :] phi_evol
	cdef np.complex128_t [:, :] phi, phi_cube, phi_sq, dW
	cdef np.complex128_t [:, :] phi_x, phi_x_cube, phi_x_sq, dW_x
	cdef Py_ssize_t n, i, j, m
	cdef double M1, M2
	cdef double kmax_half, kmax_two_thirds, kx, ky, ksq, two_pi
	cdef np.complex128_t temp
	cdef np.complex128_t birth_death, noise, mu

	M1 = 1.0
	M2 = u*(phi_s + phi_t/2.0)
	kmax_half = M_PI/2.0
	kmax_two_thirds = M_PI*2.0/3.0
	two_pi = M_PI*2
	phi_x_cube = np.empty((size, size), dtype=np.complex128)
	phi_x_sq = np.empty((size, size), dtype=np.complex128)
	dW_x = np.empty((size, size), dtype=np.complex128)


	cdef np.ndarray[np.complex128_t, ndim=2] input_forward, input_backward, output_forward, output_backward
	input_forward = pyfftw.empty_aligned((size, size), dtype='complex128')
	output_forward = pyfftw.empty_aligned((size, size), dtype='complex128')
	fft_forward = pyfftw.FFTW(input_forward, output_forward,
									direction='FFTW_FORWARD', axes=(0, 1),
									flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])
	input_backward = pyfftw.empty_aligned((size, size), dtype='complex128')
	output_backward = pyfftw.empty_aligned((size, size), dtype='complex128')
	fft_backward = pyfftw.FFTW(input_backward, output_backward,
										direction='FFTW_BACKWARD', axes=(0, 1),
										flags=['FFTW_MEASURE', 'FFTW_DESTROY_INPUT'])

	phi = init
	phi_evol = np.empty((nitr, size, size), dtype=np.float64)
	n = 0
	for i in xrange(nitr):
		input_backward[:] = phi
		phi_x = fft_backward()

		if i % batch_size == 0:
			for j in xrange(size):
				for m in xrange(size):
					phi_evol[n, j, m] = phi_x[j, m].real
			n += 1

		for j in xrange(size):
			for m in xrange(size):
				temp = phi_x[j,m]
				phi_x_sq[j,m] = temp*temp
				phi_x_cube[j,m] = temp*temp*temp

		input_forward[:] = phi_x_cube
		phi_cube = fft_forward()
		input_forward[:] = phi_x_sq
		phi_sq = fft_forward()
		input_forward[:] = np.random.normal(size=(size, size)).astype('complex128')
		dW = fft_forward()

		for j in xrange(size):
			for m in xrange(size):
				kx = fmin(j, size-j)*two_pi
				ky = fmin(m, size-m)*two_pi
				ksq = kx*kx + ky*ky
				temp = phi[j,m]
				if (kx>kmax_half) or (ky>kmax_half):
					mu = (k*ksq-a)*temp
				else:
					mu = a*phi_cube[j,m] + (k*ksq-a)*temp
				if (kx>kmax_two_thirds) or (ky>kmax_two_thirds):
					birth_death = - u*(phi_s-phi_t)*temp
				else:
					birth_death = - u*(phi_s-phi_t)*temp
				noise = sqrt(2*(M2+M1*ksq)*epsilon*dt)*dW[j,m]
				phi[j,m] = dt*(-M1*ksq*mu+birth_death) +noise + temp
		phi[0,0] = u*phi_s*phi_t*size**2 + phi[0, 0]

	return phi_evol
