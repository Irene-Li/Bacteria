import numpy as np
import pyfftw
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fmin, M_PI, ceil

@cython.wraparound(False)
@cython.boundscheck(False)
def generate_fftw_obj(np.ndarray[np.complex128_t, ndim=2] input, np.ndarray[np.complex128_t, ndim=2] output, int size, forward=True):
	input = pyfftw.empty_aligned((size, size), dtype='complex128')
	output = pyfftw.empty_aligned((size, size), dtype='complex128')
	if forward:
		dir = 'FFTW_FORWARD'
	else:
		dir = 'FFTW_BACKWARD'
	return pyfftw.FFTW(input, output, direction=dir, axes=(0, 1),
										flags=['FFTW_MEASURE']), input, output


@cython.wraparound(False)
@cython.boundscheck(False)
def evolve_sto_ps(np.complex128_t [:, :] init, double a, double k, double u, double phi_s, double phi_t, double epsilon, double dt, int nitr, int n_batches, int size):
	cdef np.float64_t [:, :, :] phi_evol
	cdef np.ndarray[np.complex128_t, ndim=2] phi, phi_cube, phi_sq, dW
	cdef np.ndarray[np.complex128_t, ndim=2] phi_x, phi_x_cube, phi_x_sq, dW_x
	cdef Py_ssize_t n, i, j, m, batch_size
	cdef double M1, M2
	cdef double kmax_half, kmax_two_thirds, kx, ky, ksq, factor
	cdef np.complex128_t temp
	cdef np.complex128_t birth_death, noise, mu

	M1 = 1.0
	M2 = u*(phi_s + phi_t/2.0)
	kmax_half = M_PI/2.0
	kmax_two_thirds = M_PI*2.0/3.0
	factor = M_PI*2.0/size

	phi_x = pyfftw.empty_aligned((size, size), dtype='complex128')
	phi = pyfftw.empty_aligned((size, size), dtype='complex128')
	phi_x_sq = pyfftw.empty_aligned((size, size), dtype='complex128')
	phi_sq = pyfftw.empty_aligned((size, size), dtype='complex128')
	phi_x_cube = pyfftw.empty_aligned((size, size), dtype='complex128')
	phi_cube = pyfftw.empty_aligned((size, size), dtype='complex128')
	dW_x = pyfftw.empty_aligned((size, size), dtype='complex128')
	dW = pyfftw.empty_aligned((size, size), dtype='complex128')

	cube_obj = pyfftw.FFTW(phi_x_cube, phi_cube, axes=(0, 1),
										flags=['FFTW_MEASURE'])
	sq_obj = pyfftw.FFTW(phi_x_sq, phi_sq, axes=(0, 1),
										flags=['FFTW_MEASURE'])
	phi_obj = pyfftw.FFTW(phi, phi_x, axes=(0, 1),
										flags=['FFTW_MEASURE'], direction='FFTW_BACKWARD')
	noise_obj = pyfftw.FFTW(dW_x, dW, axes=(0, 1),
										flags=['FFTW_MEASURE'])

	phi[:] = init

	batch_size = int(nitr/n_batches)
	phi_evol = np.empty((n_batches, size, size), dtype='float64')
	n = 0
	for i in xrange(nitr):
		phi_obj(normalise_idft=True)
		if i % batch_size == 0:
			for j in xrange(size):
				for m in xrange(size):
					phi_evol[n, j, m] = phi_x[j, m].real
			print('iteration: {},  mean: {}'.format(n, phi[0,0].real/(size*size)))
			n += 1

		for j in xrange(size):
			for m in xrange(size):
				temp = phi_x[j,m]
				phi_x_sq[j,m] = temp*temp
				phi_x_cube[j,m] = temp*temp*temp

		cube_obj.execute()
		sq_obj.execute()
		dW_x[:] = np.random.normal(size=(size, size)).astype('complex128')
		noise_obj.execute()


		for j in xrange(size):
			for m in xrange(size):
				kx = fmin(j, size-j)*factor
				ky = fmin(m, size-m)*factor
				ksq = kx*kx + ky*ky
				temp = phi[j,m]
				if (kx>kmax_half) or (ky>kmax_half):
					mu = (k*ksq-a)*temp
				else:
					mu = a*phi_cube[j,m] + (k*ksq-a)*temp
				if (kx>kmax_two_thirds) or (ky>kmax_two_thirds):
					birth_death = - u*(phi_s-phi_t)*temp
				else:
					birth_death = - u*(phi_s-phi_t)*temp - u*phi_sq[j,m]
				noise = sqrt(2*(M2+M1*ksq)*epsilon*dt)*dW[j,m]
				phi[j,m] = dt*(-M1*ksq*mu+birth_death) +noise + temp

		phi[0,0] = u*phi_s*phi_t*(size*size)*dt + phi[0,0]

	return phi_evol
