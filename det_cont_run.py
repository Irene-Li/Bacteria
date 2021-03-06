import time
import numpy as np
from PsEvolution import *

old_label = 'phi_t_-0.65_l=2'
new_label= 'phi_t_-0.65_l=2_2'

T = 5e4

start_time = time.time()
solver = PsEvolution()
solver.load(old_label)
solver.continue_evolution(T)
solver.save(new_label)
end_time = time.time()
print('The simulation took: ')
print(end_time - start_time)
