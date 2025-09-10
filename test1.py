import gymnasium as gym
import numpy as np
import random

# Initialize the Taxi environment
env = gym.make('Taxi-v3')

# Define hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.99     # Discount factor
epsilon = 0.1    # Epsilon for epsilon-greedy policy
n_episodes = 500 # Number of episodes for training

# Initialize Q-values and value functions for sub-tasks
# Q = np.zeros([env.observation_space.n, env.action_space.n])
Q = {i:[0.0001 for j in range(env.action_space.n)] for i in range(env.observation_space.n)}

V = {i:0.00001 for i in range(env.observation_space.n)}

print(Q)
# Helper function to select an action using epsilon-greedy policy
def epsilon_greedy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[10])  # Exploit

# Define a sub-task: "Drive to a location" (simple movement task)
def drive_to_location(state, target_location):
    total_reward = 0
    done = False
    while not done:
        action = epsilon_greedy(state, epsilon)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    return total_reward

# Define a sub-task: "Pick up the passenger"
def pick_up_passenger(state):
    total_reward = 0
    done = False
    while not done:
        action = epsilon_greedy(state, epsilon)
        next_state, reward, done, info,_ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    return total_reward

# Define a sub-task: "Drop off the passenger"
def drop_off_passenger(state, target_location):
    total_reward = 0
    done = False
    while not done:
        action = epsilon_greedy(state, epsilon)
        next_state, reward, done, info ,_= env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    return total_reward

# MAXQ Update Rule: Update Q-values and Value functions for sub-tasks
def maxq_update(state, action, reward, next_state, alpha, gamma):
    print(next_state)
    best_next_action = np.argmax(Q[next_state[0]])
    Q[state[0], action] += alpha * (reward + gamma * Q[next_state[0], best_next_action] - Q[state[0], action])
    V[state[0]] += alpha * (reward + gamma * V[next_state[0]] - V[state[0]])

# Training the MAXQ for Taxi-v3
# for episode in range(n_episodes):
for i in range(1):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:

        action = epsilon_greedy(state, epsilon)
        next_state, reward, done, info,_ = env.step(action)
        
        # Update Q-values for the action taken
        maxq_update(state, action, reward, next_state, alpha, gamma)
        
        total_reward += reward
        state = next_state
        
    # if episode % 50 == 0:
    #     print(f"Episode {episode}, Total Reward: {total_reward}")

# After training, let's test the agent
test_state = env.reset()
done = False
test_reward = 0
while not done:
    action = np.argmax(Q[test_state])  # Use learned policy (greedy)
    next_state, reward, done, info,_ = env.step(action)
    test_reward += reward
    test_state = next_state

print(f"Test Reward: {test_reward}")
