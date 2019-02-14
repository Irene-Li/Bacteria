import numpy as np 
from matplotlib import pyplot as plt 

phi_t = -0.5 
phi = np.arange(-2, 2, 0.01)
mu = np.arange(-1, 1, 0.01)

x, y = np.meshgrid(phi, mu)
z = - y*(x - phi_t) * x - (x**2)/2 + (x**4)/4 

plt.contourf(x, y, z)
plt.colorbar()
plt.show()