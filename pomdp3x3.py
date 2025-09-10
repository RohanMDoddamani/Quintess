import numpy as np

# 3x3 Gridworld with 9 states (3x3)
states = [(i, j) for i in range(3) for j in range(3)]

# Initial belief: uniform belief (no knowledge)
belief = {state: 1/9 for state in states}

# Transition model: Moving right
# For simplicity, assume deterministic transitions with boundary checks
def transition(state, action):
    i, j = state
    if action == "Up":
        return (max(i-1, 0), j)
    elif action == "Down":
        return (min(i+1, 2), j)
    elif action == "Left":
        return (i, max(j-1, 0))
    elif action == "Right":
        return (i, min(j+1, 2))

# Observation model: Noisy observation (probabilistic)
def observation_model(true_state, observed_state):
    # Assume 90% chance of correct observation, 10% for incorrect observation
    if true_state == observed_state:
        return 0.9
    else:
        return 0.1 / 8  # Since there are 8 other possible states

# Bayesian update after an action and observation
def update_belief(belief, action, observation):
    new_belief = {}
    # Step 1: Predict the next belief state based on action
    for state in states:
        next_state = transition(state, action)
        new_belief[next_state] = belief[state]
    
    # Step 2: Update belief based on observation
    total = 0
    for state in states:
        new_belief[state] *= observation_model(state, observation)
        total += new_belief[state]
    
    # Normalize the belief
    for state in new_belief:
        new_belief[state] /= total
    
    return new_belief

# Example action and observation
action = "Right"
observation = (1, 1)  # Noisy observation (maybe the agent isn't in (1, 1))

# Update the belief after the action and observation
new_belief = update_belief(belief, action, observation)

# Show the updated belief
for state, prob in new_belief.items():
    print(f"State {state}: {prob:.4f}")
