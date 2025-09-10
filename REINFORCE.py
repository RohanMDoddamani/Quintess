import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
# import matplotlib.pyplot as plt
import scipy as sc


# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# Select action based on policy
def select_action(policy_net, state):
    state = torch.from_numpy(state).float()
    probs = policy_net(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

# Compute discounted rewards
def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# Main training loop
def reinforce(env_name='CartPole-v1', episodes=1000, gamma=0.99, lr=1e-2):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    scores = []
    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        total_reward = 0

        done = False
        while not done:
            action, log_prob = select_action(policy_net, state)
            next_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            total_reward += reward

        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.cat(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scores.append(total_reward)
        if episode % 50 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores[-50:]):.2f}")

    env.close()

    # Plotting
    # plt.plot(scores)
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title('REINFORCE on CartPole')
    # plt.show()

# Run training
if __name__ == "__main__":
    reinforce()
