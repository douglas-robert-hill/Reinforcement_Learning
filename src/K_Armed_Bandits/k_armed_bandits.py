import numpy as np 

class kArmed_Bandits():

    def __init__(self, K: int = 10) -> None:
        """
        Initialise the K-Armed Bandit problem 

        param K : number of arms in bandit problem 
        """
        self.K = K
        

    def run(self, method: str, eps: float = 0.1, OIV: int = 5) -> None:
        """
        Run the problem by a defined method 

        param method : the 
        param eps : epsilon value when method is epsilon greedy
        param OIV : the optimistic initial value when method is OIV
        """
        if method not in ["greedy", "epsilon-greedy", "OIV", "UCB", "gradient"]:
            raise ValueError("Specified 'method' not available. Try: 'greedy', 'epsilon-greedy', 'OIV', 'UCB', 'gradient'. ")

        if method in ["greedy", "epsilon-greedy", "OIV"]:
            self.greedy_bandit()
        elif method == "UCB":
            self.ucb_bandit()
        elif method == "gradient":
            self.gradient_bandit()


    def greedy_bandit(self) -> None:
        pass


    def ucb_bandit(self) -> None:
        pass


    def gradient_bandit(self) -> None:
        pass


    def __arg_max(self, arr: np.array) -> int:
        """
        Find the optimal action from an array of expected values

        param arr : array from which optimal action should be selected 

        return max_value : integer of index of max argument - ties broken randomly 
        """
        max_value = np.where(arr == arr[np.argmax(arr)])[0].tolist()
        if len(max_value) > 1: max_value = max_value[np.random.choice(arr = len(max_value))]
        return max_value 


    def __upd_action_values(self) -> None:
        pass


    def plot_reward(self) -> None:
        pass


    def plot_optimal_action(self) -> None:
        pass


