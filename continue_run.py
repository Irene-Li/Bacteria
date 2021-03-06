import time
import numpy as np
from StoEvolution2D import *
from DetEvolution2D import *
from StoEvolution1D import *
from DetEvolution1D import *
from ActiveModelAB import * 

sto = True
twod = True
active = True 

T = 5e3

phi_t = 0.3 
u = 5e-5 
lbda = 0 
zeta = 0 

old_label = 'phi_t_{}_u_{}_lambda_{}_zeta_{}'.format(phi_t, u, lbda, zeta)
new_label= old_label + '_2'

start_time = time.time()
if twod:
    if sto:
    	if active:
    		solver = ActiveModelAB()  
    	else: 
        	solver = StoEvolution2D()
    else:
        solver = DetEvolution2D()
else:
    if sto:
        solver = StoEvolution1D()
    else:
        solver = DetEvolution1D()
solver.load(old_label)
solver.continue_evolution(T, pert=False)
solver.save(new_label)
solver.make_movie(new_label)
end_time = time.time()
print('The simulation took: ')
print(end_time - start_time)
