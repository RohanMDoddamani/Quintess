import numpy as np

# Define MDP parameters
states = [0, 1, 2, 3]  # States
actions = [0, 1]  # Actions (0 = Left, 1 = Right)
gamma = 0.9  # Discount factor
theta = 1e-6  # Convergence threshold

# Transition probabilities P(s'|s,a)
P = {
    0: {0: [(1.0, 0, 0)], 1: [(1.0, 1, 0)]},
    1: {0: [(1.0, 0, 0)], 1: [(1.0, 2, 1)]},
    2: {0: [(1.0, 1, 0)], 1: [(1.0, 3, 1)]},
    3: {0: [(1.0, 2, 0)], 1: [(1.0, 3, 0)]}
}

# Initialize policy randomly
policy = {s: np.random.choice(actions) for s in states}
V = {s: 0 for s in states}

def policy_evaluation(policy, V):
    """Evaluate the current policy."""
    while True:
        delta = 0
        for s in states:
            v = V[s]
            a = policy[s]
            V[s] = sum(p * (r + gamma * V[s_]) for p, s_, r in P[s][a])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_improvement(V):
    """Improve the policy given value function V."""
    policy_stable = True
    for s in states:
        old_action = policy[s]
        q_values = {a: sum(p * (r + gamma * V[s_]) for p, s_, r in P[s][a]) for a in actions}
        policy[s] = max(q_values, key=q_values.get)
        if old_action != policy[s]:
            policy_stable = False
    return policy, policy_stable

# Policy Iteration
while True:
    V = policy_evaluation(policy, V)
    policy, stable = policy_improvement(V)
    if stable:
        break

print("Optimal Policy:", policy)
print("Optimal Value Function:", V)
