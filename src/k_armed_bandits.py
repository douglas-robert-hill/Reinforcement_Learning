import numpy as np 
import math 
import matplotlib.pyplot as plt 

class kArmed_Bandits():

    def __init__(self, K: int = 10, steps: int = 100) -> None:
        """
        Initialise the K-Armed Bandit problem 

        param K : number of arms in bandit problem 
        param steps : number of steps to take 
        """
        self.K = K
        self.steps = steps
        self.steps_taken = 0
        self.action_values = np.ones(shape = K)
        self.action_counter = np.zeros(shape = K)
        self.reward_list = []
        self.action_list = []
        self.__set_up_dist__()


    def __set_up_dist__(self) -> None:
        """
        Set up the true distributions from which bandits will sample.
        """
        self.K_means = np.random.uniform(low = -1, high = 1, size = self.K)
        self.K_std = np.ones(shape = self.K)


    def run(self, eps: float = None, OIV: int = None, C: int = None, step_size: float = None) -> None:
        """
        Run the problem by a defined method 

        param eps : epsilon value when method is epsilon greedy
        param OIV : the optimistic initial value when method is OIV
        param C : degrees of exploration when method is UCB
        param step_size : step size for stochastic gradient ascent 
        """
        self.eps = eps
        self.OIV = OIV
        self.C = C
        self.step_size = step_size

        if self.eps is not None:
            if self.eps < 0 or self.eps > 1: 
                raise ValueError("Epsilon value should in range 0 and 1.")
        if self.OIV is not None: 
            if self.OIV < 0: 
                raise ValueError("OIV value should be greater than zero.")
        if self.C is not None:
             if self.C < 0: 
                raise ValueError("C value should be greater than zero.")
        if self.step_size is not None:
            if self.step_size < 0 or self.step_size > 1: 
                raise ValueError("step_size value should be greater than zero.")

        if self.OIV is not None: 
            self.action_values = np.full(shape = self.K, fill_value = self.OIV)

        self.greedy_bandit()


    def greedy_bandit(self) -> None:
        """
        Run Greedy Bandit with ability for Epsilon-Greedy and Optimistic Initial Values 
        """
        for _ in range(self.steps):

            self.steps_taken += 1

            if self.eps is not None and np.random.rand() < self.eps:
                action = np.random.choice(range(self.K))
            elif self.C is not None and self.steps_taken > 1:
                action = self.__arg_max_ucb__(arr = self.action_values)
            else:
                action = self.__arg_max__(arr = self.action_values)

            step_reward = self.__take_action__(action = action)
            self.action_counter[action] += 1 

            if self.step_size is not None and self.steps_taken > 1:
                self.__upd_action_value_gradients__(reward = step_reward, action = action)
            else:
                self.__upd_action_values__(reward = step_reward, action = action)

            self.reward_list.append(step_reward)
            self.action_list.append(action)


    def __take_action__(self, action: int) -> None:
        """
        Take the action by sampling from the underlying distribution 

        param action : index of action taken 
        """
        reward = np.random.normal(loc = self.K_means[action], scale = self.K_std[action])
        return float(reward)


    def __arg_max__(self, arr: np.array) -> int:
        """
        Find the optimal action from an array of expected values

        param arr : array from which optimal action should be selected 

        return max_value : integer of index of max argument - ties broken randomly 
        """
        max_value = np.where(arr == arr[np.argmax(arr)])[0].tolist()
        if len(max_value) > 1: 
            max_value = max_value[np.random.choice(range(len(max_value)))]
        else:
            max_value = int(max_value[0])
        return max_value


    def __arg_max_ucb__(self, arr: np.array) -> int:
        """
        Find the optimal action from an array of expected values under UCB

        param arr : array from which optimal action should be selected 

        return max_value : integer of index of max argument - ties broken randomly 
        """
        counter = self.action_counter
        counter[counter == 0] = 1
        arr = arr + (self.C * np.sqrt(math.log(self.steps_taken) / counter))
        max_value = np.where(arr == arr[np.argmax(arr)])[0].tolist()
        if len(max_value) > 1: 
            max_value = max_value[np.random.choice(range(len(max_value)))]
        else:
            max_value = int(max_value[0])
        return max_value


    def __upd_action_values__(self, reward: float, action: int) -> None:
        """
        Update the action value estimate based on received reward from current action

        param reward : reward received from taken action 
        param action : index of action taken 
        """
        reward_upd = self.action_values[action] + ((1 / self.action_counter[action]) * (reward - self.action_values[action]))
        self.action_values[action] = reward_upd


    def __upd_action_value_gradients__(self, reward: float, action: int) -> None:
        """
        Update the action value estimate based on stochastic gradient ascent

        param reward : reward received from taken action 
        param action : index of action taken 
        """
        avg_reward = np.mean(self.reward_list)
        probability_of_action = np.exp(self.action_values[action]) / np.sum(np.exp(self.action_values))
        reward_upd = self.action_values[action] + (self.step_size * (avg_reward - reward) * (1 - probability_of_action))
        self.action_values[action] = reward_upd


    def plot_reward(self) -> None:
        """
        Plot the average reward at each step
        """
        if self.steps_taken == 0: raise ValueError("Must run bandit before plotting.")

        x = list(range(self.steps_taken))
        y = np.cumsum(self.reward_list) / np.arange(1, (self.steps_taken + 1))

        plt.plot(x, y)
        plt.xlabel("Steps")
        plt.ylabel("Average Reward")
        plt.title("Average Reward Over Steps")
        plt.show()


    def plot_optimal_action(self) -> None:
        """
        Plot the number of times the optimal action was selected 
        """
        if self.steps_taken == 0: raise ValueError("Must run bandit before plotting.")

        x = list(range(self.steps_taken))
        optimal_action = self.__arg_max__(arr = self.K_means)
        print(optimal_action)
        action_array = np.array(self.action_list)
        dummy_for_optimal = np.where(action_array == optimal_action, 1, 0)
        y = np.cumsum(dummy_for_optimal) / np.arange(1, (len(dummy_for_optimal) + 1))

        plt.plot(x, y)
        plt.xlabel("Steps")
        plt.ylabel("% Optimal Action")
        plt.title("Optimal Action Selection Percentage Over Steps")
        plt.show()

