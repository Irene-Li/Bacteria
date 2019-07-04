import time
import numpy as np
from StoEvolutionPS import *

old_label = 'phi_t_-0.6_u_3e-05_2'
new_label= 'phi_t_-0.6_u_3e-05_3'

T = 2e4

start_time = time.time()
solver = StoEvolutionPS()
solver.load(old_label)
solver.continue_evolution(T)
solver.save(new_label)
end_time = time.time()
print('The simulation took: ')
print(end_time - start_time)
