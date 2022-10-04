
from src.monte_carlo import monteCarloAgent
import gym

env = gym.make('FrozenLake-v1')

MC_Agent = monteCarloAgent(actions = env.action_space.n, states = env.observation_space.n)

done = False
state = env.reset()

while not done:

    action = MC_Agent.step(state = state)
    state, reward, done, info = env.step(action = action)
    MC_Agent.receive_reward(reward = reward)
    env.render()

