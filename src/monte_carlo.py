
import numpy as np

class monteCarloAgent():
    """
    Implementation of every-visit monte carlo
    """

    def __init__(self, actions: int, states: int) -> None:
        """
        Initialise the Monte Carlo Learning Agent

        param actions : number of available actions 
        param states : number of available states 
        """
        self.n_actions = actions 
        self.n_states = states 
        self.steps_taken = 0
        self.action_values = np.ones(shape = self.n_actions)
        self.state_value = np.ones(shape = self.n_states)
        self.action_counter = np.zeros(shape = self.n_actions)
        self.reward_list = []
        self.action_list = []
        self.total_return = 0 
        self.current_state = None
        self.lr = 0.001


    def step(self, state: int) -> int:
        """
        Based on current state define best action 

        param state : current state

        return action : action to be taken 
        """
        self.current_state = state

        action = self._arg_max_()

        self.action_list.append(action)
        self.action_counter[action] += 1
        self.steps_taken += 1

        return action 


    def receive_reward(self, reward: float) -> None:
        """
        Receive the reward from taking a given action 

        param reward : received reward 
        """
        self._upd_state_value_(reward = reward)


    def _upd_state_value_(self, reward: float) -> None:
        """
        Update the state value functions 

        param reward : received reward 
        """
        self.state_value[self.current_state] = self.state_value[self.current_state] + (self.lr * (reward - self.state_value[self.current_state]))


    def _arg_max_(self) -> int:
        """
        Find the action to be taken by the highest expected return 
        """
        return np.argmax(0)

