import time
import numpy as np
from StoEvolutionPS import *

old_label = 'phi_t_-0.65_u_5e-05_flat_2'
new_label= 'phi_t_-0.65_u_5e-05_flat_3'
path ='/store/SOFT/yl511/PS/'

T = 1e4

start_time = time.time()
solver = StoEvolutionPS()
solver.load(old_label)
solver.continue_evolution(T)
solver.save(new_label, path=path)
end_time = time.time()
print('The simulation took: ')
print(end_time - start_time)
