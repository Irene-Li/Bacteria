from FE import *

label = 'dx_1_dt_1'

t_size = 100 
n_samples = 3


solver = FeEvolution()
solver.load(label)
# solver.plot_evolution(t_size, label)

solver.plot_average(label)
# solver.compute_mu(label, n=n_samples)
solver.plot_phi_bar_dot(label)

# solver.plot_phase_space(label)
solver.plot_samples(label, n=n_samples)