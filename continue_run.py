import time
import numpy as np
from FdEvolution import * 

old_label = 'small_box_u_1e-6'
new_label= 'small_box_u_1e-6_2'

dt = 1e-4
T = 1e6
n_steps = int(T/dt)


start_time = time.time()
solver = FdEvolution()
solver.load(old_label)
solver.continue_evolution(n_steps)
solver.save(new_label)
end_time = time.time()
print('The simulation took: ')
print(end_time - start_time)



