
from src.k_armed_bandits import * 

greedy = kArmed_Bandits(K = 10, steps = 2000)
greedy.run()
greedy.plot_optimal_action()

eps_greedy = kArmed_Bandits(K = 10, steps = 2000)
eps_greedy.run(eps = 0.1)
eps_greedy.plot_optimal_action()

oiv_greedy = kArmed_Bandits(K = 10, steps = 2000)
oiv_greedy.run(OIV = 4)
oiv_greedy.plot_optimal_action()

ucb_greedy = kArmed_Bandits(K = 10, steps = 2000)
ucb_greedy.run(C = 2)
ucb_greedy.plot_optimal_action()

stochastic_greedy = kArmed_Bandits(K = 10, steps = 2000)
stochastic_greedy.run(step_size = 0.1)
stochastic_greedy.plot_optimal_action()