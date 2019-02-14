import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import * 

length = 50
dx = 0.5

x = np.arange(0, length, dx)
phi = np.cos(10*np.pi*x/length) + np.cos(20*np.pi*x/length)

phi_ft = rfft(phi)
S = np.zeros(int(length/dx/2 + 1))
S[0] = phi_ft[0]**2
S[-1] = phi_ft[0]**2 
S[1:-1] = (np.roll(phi_ft, -1)**2 + phi_ft**2)[1:-1:2]

q = np.arange(int(length/dx/2+1))/(length) * 2 * np.pi 

plt.plot(q, S, 'k-')
plt.show()