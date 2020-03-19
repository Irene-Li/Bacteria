import numpy as np
from scipy.special.cython_special cimport jv
from scipy.optimize import minimize as scipymin
cimport numpy as np
cimport cython
from cython.view cimport array
from libc.math cimport sqrt, fmin, M_PI, pow, sinh, exp, INFINITY

cdef class FreeEnergy(object):
  cdef double a, b, k, c, m, a_eff, sigma, lattice_scale_factor, l_int
  cdef int Nterms, lattice_sym

  def __init__(self, double a, double b, double k, double c, double m):
    self.a = a
    self.b = b
    self.k = k
    self.c = c
    self.m = m
    self.a_eff = a + k*m*m
    self.sigma = sqrt(self.a_eff*k/18)
    self.l_int = 2*sqrt(2*self.k/self.a)

  def set_m_phit(self, double m, double phit):
    self.m = m
    self.c = - self.b*phit**3

  cpdef double phit(self):
    return -pow(self.c/self.b, 1.0/3.0)

  def initialise(self, int Nterms):
    pass

  @cython.cdivision(True)
  cpdef double free_energy_for_uniform(self):
    phi_t = - pow(self.c/self.b, 1.0/3.0)
    f = self._f(phi_t)
    f_nl = self.a_eff*phi_t*phi_t/2.0
    return f + f_nl

  @cython.wraparound(False)
  @cython.boundscheck(False)
  def minimise_lattice(self, double Lmax):
    fun_lat = lambda x: self.free_energy_for_lattice(x[0], x[1], x[2])
    x0 = [Lmax/2, 1, -1]
    res_lat = scipymin(fun_lat, x0, method='Nelder-Mead', options={'fatol':1e-5, 'maxiter':1e3})
    f_lat = res_lat.fun
    # if not res_lat.success:
    #   print(res_lat.message, self.phit(), self.m)
    return f_lat

  @cython.wraparound(False)
  @cython.boundscheck(False)
  def minimise_lamellar(self, double Lmax):
    fun_lam = lambda x: self.free_energy_for_lamellar(x[0], x[1], x[2])
    x0 = [Lmax/2, 1, -1]
    res_lam = scipymin(fun_lam, x0, method='Nelder-Mead', options={'fatol':1e-5, 'maxiter':1e3})
    f_lam = res_lam.fun
    # if not res_lam.success:
    #   print(res_lam.message, self.phit(), self.m)
    return f_lam



  @cython.wraparound(False)
  @cython.boundscheck(False)
  def minimise_over_AB(self, double Lmax, int N):
    cdef double phit
    cdef double [:, :] minimums
    cdef double [:] As, Bs
    phit = self.phit()
    As = np.linspace(-phit, 1-phit, N)
    Bs = np.linspace(-1+phit, phit, N)
    minimums = array(shape=(N,N), itemsize=sizeof(np.float64_t), format='d')
    for (i, A) in enumerate(As):
      for (j, B) in enumerate(Bs):
        minimums[j, i] = self.min_free_energy_lattice(Lmax, A, B)
    np.save("minimums.npy", minimums)


  def min_free_energy_lattice(self, double Lmax, double A, double B):
    cdef double Lmin
    Lmin = 2*sqrt(self.k/self.a)
    fun = lambda x: self.free_energy_for_lattice(x[0], A, B)
    res = scipymin(fun, [Lmax/2], method='Nelder-Mead', options={'fatol':1e-5, 'maxiter':1e3})
    return res.fun

  @cython.wraparound(False)
  @cython.boundscheck(False)
  def total_free_energy_for_range(self, double Lmax, double A, double B):
    cdef np.float64_t [:] free_energy
    cdef int i
    cdef double deltaL, r, L
    deltaL = Lmax/1000.0
    r = self._ratio(A, B)
    free_energy = array(shape=(1000,), itemsize=sizeof(np.float64_t), format='d')
    for i in range(1, 1001):
      L = deltaL * i
      free_energy[i] = self.free_energy_for_lattice(L, A, B)
    return free_energy

  @cython.wraparound(False)
  @cython.boundscheck(False)
  def free_energy_for_lamellar_range(self, double Lmax, double A, double B):
    cdef np.float64_t [:] free_energy
    cdef double deltaL, L
    cdef int i
    deltaL = Lmax/1000.0
    free_energy = array(shape=(1000,), itemsize=sizeof(np.float64_t), format='d')
    for i in range(1, 1001):
      L = deltaL * i
      free_energy[i] = self.free_energy_for_lamellar(L, A, B)
    return free_energy


  @cython.cdivision(True)
  cdef double free_energy_for_lamellar(self, double L, double A, double B):
    cdef double r, phit
    phit = self.phit()
    r = self._ratio_lamellar(A, B)
    if r*L < self.l_int or B > phit or A < phit:
      print(r*L, self.l_int)
      return INFINITY
    cdef double a, f_loc, temp, f_nl
    a = self.m*L/2.0
    f_loc = self._f(A)*r + self._f(B)*(1.0-r)
    f_loc += 2.0*self.sigma*((A-B)**2.0)/L
    temp = (A-B)*sinh(a*(1.0-r))*sinh(a*r)/(a*sinh(a))
    f_nl = A*(A*r - temp) + B*(B*(1.0-r) + temp)
    f_nl *= self.a_eff/2
    return f_loc + f_nl

  cdef double free_energy_for_lattice(self, double L, double A, double B):
    cdef double r, phit
    phit = self.phit()
    r = self._ratio(A, B)
    if r*L < self.l_int or B > phit or A < phit:
      print(r*L, self.l_int)
      return INFINITY
    return self.free_energy_local(L, A, B, r) + self.free_energy_nl(L, A, B, r)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.cdivision(True)
  cdef double free_energy_nl(self, double L, double A, double B, double r):
    cdef double chi, sum, area_ratio, term1, term2
    cdef Py_ssize_t nx, ny
    chi = self.m*L*self.lattice_scale_factor/(2.0*M_PI)
    sum = 0.0
    for nx in range(1, self.Nterms):
      for ny in range(0, self.Nterms):
        n = self._distance(<double>nx, <double>ny)
        Ek = self._Ek(n, r, chi)
        sum += Ek
    term1 = self.lattice_sym*(chi*r)**2*sum*(A-B)**2
    area_ratio = M_PI/self.lattice_scale_factor*r*r
    term2 = (A*area_ratio+B*(1.0-area_ratio))**2
    return self.a_eff*(term1+term2)/2.0

  @cython.cdivision(True)
  cdef double free_energy_local(self, double L, double A, double B, double r):
    cdef double area_ratio, term1
    area_ratio = M_PI/self.lattice_scale_factor*r*r
    term1 = self._f(A)*area_ratio + self._f(B)*(1-area_ratio)
    term1 += self.sigma*((A-B)**2)*2.0*M_PI/self.lattice_scale_factor*r/L
    return term1

  @cython.cdivision(True)
  cdef double _f(self, double x):
    return self.c*x - self.a_eff*x*x/2 + self.b*pow(x, 4)/4

  @cython.cdivision(True)
  @cython.nonecheck(False)
  cdef double _Ek(self, double n, double ratio, double chi):
    cdef double term1, term2
    term1 = jv(1, ratio*n*2.0*M_PI/self.lattice_scale_factor)/n
    term2 = chi*chi + n*n
    return term1*term1/term2

  @cython.cdivision(True)
  cdef double _ratio_lamellar(self, double A, double B):
      return (-self.c-self.b*B**3)/(self.b*(A**3-B**3))

  cdef double _ratio(self, double A, double B):
      pass #to implement in subclasses

  cdef double _distance(self, double nx, double ny):
      pass #to impletement in subclass



cdef class FreeEnergySquare(FreeEnergy):

  def initialise(self, int Nterms):
    self.Nterms = Nterms
    self.lattice_scale_factor = 1
    self.lattice_sym = 4

  # @cython.cdivision(True)
  cdef double _ratio(self, double A, double B):
    return sqrt((- self.c - self.b*B**3)/(self.b*A**3 - self.b*B**3)/M_PI)

  cdef double _distance(self, double nx, double ny):
    return sqrt(nx*nx+ny*ny)


cdef class FreeEnergyTriangle(FreeEnergy):

  def initialise(self, int Nterms):
    self.Nterms = Nterms
    self.lattice_scale_factor = sqrt(3)/2
    self.lattice_sym = 6

  @cython.cdivision(True)
  cdef double _ratio(self, double A, double B):
    return sqrt(sqrt(3)*(-self.c - self.b*B**3)/(2*M_PI*self.b*(A**3-B**3)))

  cdef double _distance(self, double nx, double ny):
    return sqrt(nx*nx+ny*ny+nx*ny)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def phases(FreeEnergy obj_sq, FreeEnergy obj_tri, double logm_min, double logm_max, int N):
  cdef int [:, :] phases
  cdef Py_ssize_t i, j
  cdef double logm, phit, delta_logm, delta_phit, m, uniform, f_sq, f_tri, f_lam, Lmax
  phases = array(shape=(N,N), itemsize=sizeof(int), format='i')
  logm = logm_min
  delta_logm = (logm_max - logm_min)/(N-1)
  delta_phit = 1.19/(N-1)
  for i in range(N):
    phit = -1.2
    m = exp(logm)
    print(i)
    for j in range(N):
      obj_sq.set_m_phit(m, phit)
      obj_tri.set_m_phit(m, phit)
      Lmax = 10.0/m
      f_sq = obj_sq.minimise_lattice(Lmax)
      f_tri = obj_tri.minimise_lattice(Lmax)
      f_lam = obj_tri.minimise_lamellar(Lmax)
      uniform = obj_tri.free_energy_for_uniform()
      phases[j, i] = np.argmin([f_sq, uniform, f_tri, f_lam])
      phit += delta_phit
    logm += delta_logm
  np.save("phases.npy", phases)
