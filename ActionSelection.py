import numpy as np
import random

# Define grid size and actions
GRID_SIZE = 2
ACTIONS = ['up', 'down', 'left', 'right']
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
TAU = 1.0  # Initial temperature
TAU_DECAY = 0.99  # Decay factor for temperature
EPISODES = 1000  # Number of episodes

# Define rewards
REWARDS = np.full((GRID_SIZE, GRID_SIZE), -1)  # Default reward
REWARDS[1, 1] = 10  # Goal reward

# Initialize Q-table
Q_table = np.random.uniform((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
EPSILON = 0.2  # Exploration rate


States = [1,3,2,4]

print(Q_table)

def softmax_action(state, tau):
    exp_q = np.exp(state)  # Apply softmax formula
    probabilities = exp_q / np.sum(exp_q)  # Normalize
    print(probabilities)
    print(sum(probabilities))
    print(np.random.choice(ACTIONS, p=probabilities) )
    return np.random.choice(ACTIONS, p=probabilities)  # Select action based on probabilities

softmax_action(States,TAU)

def choose_action(state):
    if random.uniform(0, 1) < EPSILON:  # Explore
        return random.choice(ACTIONS)
    else:  # Exploit
        return ACTIONS[np.argmax(Q_table[state[0], state[1]])]