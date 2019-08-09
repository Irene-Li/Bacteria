import time
import numpy as np
from StoEvolution2D import *
from DetEvolution2D import *
from StoEvolution1D import *
from DetEvolution1D import *

sto = False
twod = True

old_label = 'det_phi_t_0_delta_0.1_3'
new_label= 'det_phi_t_0_delta_0.1_4'

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
end_time = time.time()
print('The simulation took: ')
print(end_time - start_time)
