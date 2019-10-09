import numpy as np
import mkl_fft
cimport numpy as np
cimport cython
from cython.view cimport array
from libc.math cimport sqrt, fmin, M_PI, ceil

@cython.wraparound(False)
@cython.boundscheck(False)
def delta_phi_ps_1d(np.complex128_t[:] phi_x, double a, double k, double u, double phi_s, double phi_t, double kmax_half, double kmax_two_thirds, double factor, int size):
	cdef np.complex128_t [:] phi, phi_cube, phi_sq, delta_phi
	cdef np.complex128_t [:] phi_x_cube, phi_x_sq
	cdef Py_ssize_t j
	cdef double kx, ksq
	cdef np.complex128_t temp
	cdef np.complex128_t birth_death, mu

	phi_x_sq = array(shape=(size,), itemsize=sizeof(np.complex128_t), format='Zd')
	phi_x_cube = array(shape=(size,), itemsize=sizeof(np.complex128_t), format='Zd')
	delta_phi = array(shape=(size,), itemsize=sizeof(np.complex128_t), format='Zd')
	phi = mkl_fft.fft(phi_x)

	for j in xrange(size):
		temp = phi_x[j]
		phi_x_sq[j] = temp*temp
		phi_x_cube[j] = temp*temp*temp

	phi_cube = mkl_fft.fft(phi_x_cube)
	phi_sq = mkl_fft.fft(phi_x_sq)

	for j in xrange(size):
		kx = fmin(j, size-j)*factor
		ksq = kx*kx
		temp = phi[j]
		if (kx>kmax_half):
			mu = (k*ksq-a)*temp
		else:
			mu = a*phi_cube[j] + (k*ksq-a)*temp
		if (kx>kmax_two_thirds):
			birth_death = - u*(phi_s-phi_t)*temp
		else:
			birth_death = - u*(phi_s-phi_t)*temp - u*phi_sq[j]
		delta_phi[j] = -ksq*mu+birth_death

	delta_phi[0] = u*phi_s*phi_t*(size) + delta_phi[0]

	return mkl_fft.ifft(delta_phi)



@cython.wraparound(False)
@cython.boundscheck(False)
def evolve_sto_ps_1d(np.complex128_t[:] init, double a, double k, double u, double phi_s, double phi_t, double epsilon, double dt, int nitr, int n_batches, int size):
	cdef np.float64_t [:, :] phi_evol
	cdef np.complex128_t [:] phi, phi_cube, phi_sq, dW
	cdef np.complex128_t [:] phi_x, phi_x_cube, phi_x_sq
	cdef Py_ssize_t n, i, j, batch_size
	cdef double M1, M2
	cdef double kmax_half, kmax_two_thirds, kx, ksq, factor
	cdef np.complex128_t temp
	cdef np.complex128_t birth_death, noise, mu

	M1 = 1.0
	M2 = u*(phi_s + phi_t/2.0)
	kmax_half = M_PI/2.0
	kmax_two_thirds = M_PI*2.0/3.0
	factor = M_PI*2.0/size

	phi_x_sq = array(shape=(size,), itemsize=sizeof(np.complex128_t), format='Zd')
	phi_x_cube = array(shape=(size,), itemsize=sizeof(np.complex128_t), format='Zd')

	phi = init
	batch_size = int(nitr/n_batches)
	nitr = batch_size * n_batches
	phi_evol = np.empty((n_batches, size), dtype='float64')
	n = 0
	for i in xrange(nitr):
		phi_x = mkl_fft.ifft(phi)

		if i % batch_size == 0:
			for j in xrange(size):
				phi_evol[n, j] = phi_x[j].real
			print('iteration: {},  mean: {}'.format(n, phi[0].real/size))
			n += 1

		for j in xrange(size):
			temp = phi_x[j]
			phi_x_sq[j] = temp*temp
			phi_x_cube[j] = temp*temp*temp

		phi_cube = mkl_fft.fft(phi_x_cube)
		phi_sq = mkl_fft.fft(phi_x_sq)
		dW = mkl_fft.fft(np.random.normal(size=(size)).astype('complex128'))

		for j in xrange(size):
			kx = fmin(j, size-j)*factor
			ksq = kx*kx
			temp = phi[j]
			if (kx>kmax_half):
				mu = (k*ksq-a)*temp
			else:
				mu = a*phi_cube[j] + (k*ksq-a)*temp
			if (kx>kmax_two_thirds):
				birth_death = - u*(phi_s-phi_t)*temp
			else:
				birth_death = - u*(phi_s-phi_t)*temp - u*phi_sq[j]
			noise = sqrt(2*(M2+M1*ksq)*epsilon*dt)*dW[j]
			phi[j] = dt*(-M1*ksq*mu+birth_death) +noise + temp

		if i % batch_size == 0:
			for j in xrange(size):
				phi_evol[n, j] = phi_x[j].real
			print('iteration: {},  mean: {}'.format(n, phi[0].real/size))
			n += 1

		phi[0] = u*phi_s*phi_t*(size*size)*dt + phi[0]

	return phi_evol


@cython.wraparound(False)
@cython.boundscheck(False)
def evolve_sto_ps(np.complex128_t [:, :] init, double a, double k, double u, double phi_s, double phi_t, double epsilon, double dt, int nitr, int n_batches, int size):
	cdef np.float64_t [:, :, :] phi_evol
	cdef np.complex128_t [:, :] phi, phi_cube, phi_sq, dW
	cdef np.complex128_t [:, :] phi_x, phi_x_cube, phi_x_sq
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

	phi_x_sq = array(shape=(size, size), itemsize=sizeof(np.complex128_t), format='Zd')
	phi_x_cube = array(shape=(size, size), itemsize=sizeof(np.complex128_t), format='Zd')

	phi = init
	batch_size = int(nitr/n_batches)
	nitr = batch_size * n_batches
	phi_evol = np.empty((n_batches, size, size), dtype='float64')
	n = 0
	for i in xrange(nitr):
		phi_x = mkl_fft.ifft2(phi)

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

		phi_cube = mkl_fft.fft2(phi_x_cube)
		phi_sq = mkl_fft.fft2(phi_x_sq)
		dW = mkl_fft.fft2(np.random.normal(size=(size, size)).astype('complex128'))

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

@cython.wraparound(False)
@cython.boundscheck(False)
def evolve_det_ps(np.complex128_t [:, :] init, double a, double k, double u, double phi_s, double phi_t, double dt, int nitr, int n_batches, int size):
	cdef np.float64_t [:, :, :] phi_evol
	cdef np.complex128_t [:, :] phi, phi_cube, phi_sq
	cdef np.complex128_t [:, :] phi_x, phi_x_cube, phi_x_sq
	cdef Py_ssize_t n, i, j, m, batch_size
	cdef double kmax_half, kmax_two_thirds, kx, ky, ksq, factor
	cdef np.complex128_t temp
	cdef np.complex128_t birth_death, mu

	kmax_half = M_PI/2.0
	kmax_two_thirds = M_PI*2.0/3.0
	factor = M_PI*2.0/size

	phi_x_sq = array(shape=(size, size), itemsize=sizeof(np.complex128_t), format='Zd')
	phi_x_cube = array(shape=(size, size), itemsize=sizeof(np.complex128_t), format='Zd')

	phi = init
	batch_size = int(nitr/n_batches)
	nitr = batch_size * n_batches
	phi_evol = np.empty((n_batches, size, size), dtype='float64')
	n = 0
	for i in xrange(nitr):
		phi_x = mkl_fft.ifft2(phi)

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

		phi_cube = mkl_fft.fft2(phi_x_cube)
		phi_sq = mkl_fft.fft2(phi_x_sq)

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
				phi[j,m] = dt*(-ksq*mu+birth_death) + temp

		phi[0,0] = u*phi_s*phi_t*(size*size)*dt + phi[0,0]

	return phi_evol
