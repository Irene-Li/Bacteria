import numpy as np
from scipy.special.cython_special cimport jv
cimport numpy as np
cimport cython
from cython.view cimport array
from libc.math cimport sqrt, fmin, M_PI, pow, sinh

cdef class FreeEnergy(object):
  cdef double a, b, k, c, m, a_eff, sigma, lattice_scale_factor
  cdef int Nterms, lattice_sym

  def __init__(self, double a, double b, double k, double c, double m):
    self.a = a
    self.b = b
    self.k = k
    self.c = c
    self.m = m
    self.a_eff = a + k*m*m
    self.sigma = sqrt(self.a_eff*k/18)

  def set_m_phit(self, double m, double phit):
    self.m = m
    self.c = - self.b*phit**3

  def initialise(self, int Nterms):
    pass

  def free_energy_for_uniform(self):
    phi_t = - pow(self.c/self.b, 1.0/3.0)
    f = self._f(phi_t)
    f_nl = self.a_eff*phi_t*phi_t/2.0
    return f + f_nl

  @cython.wraparound(False)
  @cython.boundscheck(False)
  def phases(self, Lmax):
    cdef double [:] logms, ms, phits

  @cython.wraparound(False)
  @cython.boundscheck(False)
  cpdef (double, double) minimise(self, double Lmax, double phi_t):
    cdef double diff, lat_min, lam_min, f_lat, f_lam, A, B
    diff = fmin(phi_t+1, 0.1)
    lat_min = 0
    lam_min = 0
    for A in [1-diff, 1, 1+diff]:
      for B in [-1-diff, -1, -1+diff]:
        f_lat = self.min_free_energy_lattice(Lmax, 1, -1)
        f_lam = self.min_free_energy_laminar(Lmax, 1, -1)
        lat_min = fmin(f_lat, lat_min)
        lam_min = fmin(f_lam, lam_min)
    return (lat_min, lam_min)

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
      free_energy[i] = self.free_energy_local(L, A, B, r)+self.free_energy_nl(L, A, B, r)
    return free_energy

  @cython.wraparound(False)
  @cython.boundscheck(False)
  def free_energy_for_laminar_range(self, double Lmax, double A, double B):
    cdef np.float64_t [:] free_energy
    cdef double deltaL, r, L
    cdef int i
    deltaL = Lmax/1000.0
    r = self._ratio_laminar(A, B)
    free_energy = array(shape=(1000,), itemsize=sizeof(np.float64_t), format='d')
    for i in range(1, 1001):
      L = deltaL * i
      free_energy[i] = self.free_energy_for_laminar(L, A, B, r)
    return free_energy


  @cython.cdivision(True)
  cdef double free_energy_for_laminar(self, double L, double A, double B, double r):
    cdef double a, f_loc, temp, f_nl
    a = self.m*L/2.0
    f_loc = self._f(A)*r + self._f(B)*(1.0-r)
    f_loc += 2.0*self.sigma*((A-B)**2.0)/L
    temp = (A-B)*sinh(a*(1.0-r))*sinh(a*r)/(a*sinh(a))
    f_nl = A*(A*r - temp) + B*(B*(1.0-r) + temp)
    f_nl *= self.a_eff/2.0
    return f_loc + f_nl

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.cdivision(True)
  cdef double free_energy_nl(self, double L, double A, double B, double r):
    cdef double chi, sum, area_ratio, term1, term2
    cdef int nx, ny
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

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.cdivision(True)
  cdef double min_free_energy_lattice(self, double Lmax, double A, double B):
    cdef double r, minimum, f, deltaL, L
    cdef int i
    deltaL = Lmax/1000.0
    r = self._ratio(A, B)
    minimum = 0.0
    for i in range(1, 1001):
      L = deltaL*i
      f = self.free_energy_local(L, A, B, r)
      f+= self.free_energy_nl(L, A, B, r)
      minimum = fmin(f, minimum)
    return minimum

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.cdivision(True)
  cdef double min_free_energy_laminar(self, double Lmax, double A, double B):
    cdef double min, r, f, deltaL, L
    cdef int i
    deltaL = Lmax/1000.0
    r = self._ratio_laminar(A, B)
    min = 0.0
    for i in range(1, 1001):
      L = deltaL*i
      f = self.free_energy_for_laminar(L, A, B, r)
      min = fmin(min, f)
    return min

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
  cdef double _ratio_laminar(self, double A, double B):
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

  @cython.cdivision(True)
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
