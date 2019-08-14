import time
import numpy as np
from StoEvolution2D import *
from DetEvolution2D import *
from StoEvolution1D import *
from DetEvolution1D import *

sto = False
twod = True

old_label = 'phi_t_-0.5_u_0.0002_4'
new_label= 'phi_t_-0.5_u_0.0002_5'

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
solver.continue_evolution(T)
solver.save(new_label)
solver.make_movie(new_label)
end_time = time.time()
print('The simulation took: ')
print(end_time - start_time)
