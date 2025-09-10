import gymnasium as gym
import numpy as np
from collections import defaultdict

# Environment
env = gym.make("FrozenLake-v1", is_slippery=True)  # deterministic for simplicity

# Parameters
alpha = 0.1         # learning rate
gamma = 0.99       # discount factor
num_episodes = 5000


V = {i:0.00001 for i in range(16)}
Q = {i:[0.0001 for j in range(4)] for i in range(env.observation_space.n)}

def policy(state):
    return np.random.choice([1, 2])  # down or right


def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])
# TD(0) algorithm
for episode in range(2000):
    state = env.reset()[0]  # Gym v0.26+ returns (obs, info)
    done = False

    while not done:
        # print(state)
        action = epsilon_greedy_policy(state,0.1)
        next_state, reward, done, truncated, info = env.step(action)
        # print(next_state)
        # TD(0) update
        next_action = epsilon_greedy_policy(next_state,0.1)
        # V[state] += alpha * (reward + gamma * V[next_state] - V[state])
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][next_action] - Q[state][action]
        )
        
        state = next_state


# print(Q)
# Display learned value function
# for s in range(env.observation_space.n):
#     print(f"V({s}) = {Q[s]}")

VN = {i:0 for i in range(env.observation_space.n)}
for state in range(env.observation_space.n):
    VN[state] = np.max(Q[state])  # value under greedy policy

# print(VN)

for s in range(env.observation_space.n):
    print(f"V({s}) = {VN[s]}")