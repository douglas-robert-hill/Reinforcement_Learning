
from src.K_Armed_Bandits.k_armed_bandits import * 

a = kArmed_Bandits(K = 10, steps = 2000)
a.run(method = "epsilon-greedy", eps = 0.1, OIV = 1)
a.plot_optimal_action()