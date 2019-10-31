import time
import numpy as np
from StoEvolution2D import *
from DetEvolution2D import *
from StoEvolution1D import *
from DetEvolution1D import *

sto = True
twod = True

phi_t = -0.4
phi_s = 1.4
u = 1e-5
old_label = 'phi_t_{}_phi_s_{}_u_{}'.format(phi_t, phi_s, u)
new_label= 'phi_t_{}_phi_s_{}_u_{}_2'.format(phi_t, phi_s, u)

T = 5e3

start_time = time.time()
if twod:
    if sto:
        solver = StoEvolution2D()
    else:
        solver = DetEvolution2D()
else:
    if sto:
        solver = StoEvolution1D()
    else:
        solver = DetEvolution1D()
solver.load(old_label)
solver.continue_evolution(T, pert=True)
solver.save(new_label)
solver.make_movie(new_label)
end_time = time.time()
print('The simulation took: ')
print(end_time - start_time)
