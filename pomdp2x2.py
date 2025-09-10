import numpy as np

# States (row, col)
states = [(0,0), (0,1), (1,0), (1,1)]
n_states = len(states)

# Initialize uniform belief
belief = np.full(n_states, 1.0 / n_states)

# Observation model: P(o | s)
def observation_model(state, obs):
    if obs == "Wall-North":
        if state[0] == 0:  # top row has north wall
            return 0.9
        else:
            return 0.1
    return 0.5

# Transition model: P(s' | s, a)
def transition_model(prev_state, action):
    next_states = []
    for s in states:
        prob = 0.0
        if action == "Up":
            if prev_state[0] > 0 and (prev_state[0]-1, prev_state[1]) == s:
                prob = 0.9
            elif s == prev_state:
                prob = 0.1
        next_states.append(prob)
    return np.array(next_states)

# Belief update
def update_belief(belief, action, observation):
    prior = np.zeros(n_states)
    for i, s_prime in enumerate(states):
        for j, s in enumerate(states):
            T = transition_model(s, action)[i]
            prior[i] += T * belief[j]

    # Update with observation
    new_belief = np.zeros(n_states)
    for i, s in enumerate(states):
        new_belief[i] = observation_model(s, observation) * prior[i]

    # Normalize
    new_belief /= np.sum(new_belief)
    return new_belief

# Example belief update
belief = update_belief(belief, action="Up", observation="Wall-North")
print("Updated Belief:", belief)





# Solving a **Partially Observable Markov Decision Process (POMDP)** using **Bayesian belief** is a challenging but powerful approach. Let's break it down.

# ### 1. **POMDP Overview**

# A **POMDP** is an extension of a standard Markov Decision Process (MDP), but it handles environments where the agent cannot fully observe the state. Instead, the agent has to make decisions based on **partial information**, often represented as a belief state (a probability distribution over possible states).

# A POMDP is defined by:

# * **States** $S$: A set of possible states in the environment.
# * **Actions** $A$: A set of actions the agent can take.
# * **Observations** $O$: A set of observations the agent receives after taking actions.
# * **Transition function** $T(s'|s, a)$: Probability of transitioning from state $s$ to state $s'$ given action $a$.
# * **Observation function** $Z(o|s', a)$: Probability of receiving observation $o$ after transitioning to state $s'$ and taking action $a$.
# * **Reward function** $R(s, a)$: Reward the agent receives after taking action $a$ in state $s$.
# * **Belief state** $b$: A probability distribution over states, reflecting the agent's uncertainty about which state it's in.

# ### 2. **Bayesian Belief in POMDP**

# The core idea behind using **Bayesian belief** in POMDPs is that the agent uses its observations and a **Bayesian update** to refine its belief state over time. The belief state $b_t$ is updated each time an action is taken and an observation is received.

# #### Bayesian Update

# Let’s assume that the agent is in some belief state $b_t$ at time $t$, and it performs an action $a_t$ and receives an observation $o_t$. The belief state is updated as follows:

# 1. **Predict the next belief** (based on the previous belief and the action):

#    $$
#    b'(s') = \sum_{s \in S} T(s' | s, a_t) b_t(s)
#    $$

#    This gives the predicted belief about the next state before observing anything.

# 2. **Update the belief** based on the new observation $o_t$ using **Bayes' rule**:

#    $$
#    b_{t+1}(s') = \eta \cdot Z(o_t | s', a_t) \cdot b'(s')
#    $$

#    Here, $\eta$ is a normalization factor ensuring that the belief state sums to 1.

# 3. **Normalization factor** $\eta$ is given by:

#    $$
#    \eta = \frac{1}{\sum_{s' \in S} Z(o_t | s', a_t) \cdot b'(s')}
#    $$

#    After this update, $b_{t+1}(s')$ represents the agent's new belief about being in state $s'$ after taking action $a_t$ and receiving observation $o_t$.

# ### 3. **Decision Making in POMDP**

# Once the belief state has been updated, the agent needs to choose an optimal action based on the current belief. This can be done using **Value Iteration** or **Policy Iteration**, but adapted to the belief space rather than the state space. The key steps are:

# * **Belief MDP**: Treat the belief states as a new state space and apply standard dynamic programming methods to optimize decisions.
# * **Value Function**: The value function $V(b)$ represents the expected cumulative reward starting from belief state $b$. The goal is to maximize the expected reward.

# ### 4. **Planning under Uncertainty**

# To make a decision, you need to compute the expected reward of taking each possible action given the current belief state, as well as how the belief will evolve after observing the outcome of an action. This requires solving a **belief MDP**:

# 1. **Action Selection**: Choose an action $a$ that maximizes the expected future reward, taking into account the updated belief state after observation.

# 2. **Recursion**: Use recursive techniques to solve for the expected reward over time, as the belief state evolves.

# ### 5. **Approximation Methods**

# Exact solutions to POMDPs can be computationally expensive. To make solving POMDPs feasible in practice, several **approximation methods** are used:

# * **Point-based Value Iteration (PBVI)**: A method that approximates the solution by focusing on a limited set of belief points.
# * **Monte Carlo Sampling**: Randomly sampling from the belief space and updating values based on these samples.
# * **QLearning in POMDPs**: Reinforcement learning techniques such as Q-learning can be adapted for POMDPs by considering belief states as part of the state space.

# ### 6. **Applications**

# POMDPs and Bayesian belief updating are especially useful in scenarios like:

# * **Robotics**: Where a robot may not know its exact position (localization) and has to infer it from noisy sensors.
# * **Autonomous Vehicles**: Making decisions under partial observability (e.g., incomplete information about the environment).
# * **Medical Diagnosis**: Where the doctor has partial information about the patient’s condition.

# ### Summary

# * **Bayesian belief** helps in updating the agent’s belief about the state of the world based on its observations.
# * The process involves predicting the belief after an action and then updating it based on the new observation.
# * Decision making is done by treating the belief states as a new state space and optimizing the decision process.
# * Exact solutions are hard, so **approximation methods** like **Point-based Value Iteration** are commonly used.

# Would you like a more detailed explanation on any of these aspects or perhaps a concrete example to work through?
