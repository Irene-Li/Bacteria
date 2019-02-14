import time
import numpy as np
from PDE import *

# the label for this run 
label = 'a_3e-2'
slow_mfold_label = '../slow_mfold/init_0.7/init_70'

# Set the run paramters 
dt = 6e-3
T = dt * 1e7
n_batches = int(1e2)
u = 5e-5 

start_time = time.time()
solver = FdEvolution()
solver.load(slow_mfold_label)
solver.u = u 
solver.continue_evolution(T, dt, n_batches)
solver.save(label)
end_time = time.time()
print('The simulation took: ')
print(end_time - start_time)