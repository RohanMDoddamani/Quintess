import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque

# Hyperparameters
gamma = 0.99          # discount factor
epsilon = 1.0         # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
memory_size = 100000

# Environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Experience replay memory
memory = deque(maxlen=memory_size)

# Build Q-network model
def build_model():
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=(state_size,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model

model = build_model()

# Epsilon-greedy action selection
def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    q_values = model.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])

# Replay experience and train
def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    states = np.array([m[0] for m in minibatch])
    actions = np.array([m[1] for m in minibatch])
    rewards = np.array([m[2] for m in minibatch])
    next_states = np.array([m[3] for m in minibatch])
    dones = np.array([m[4] for m in minibatch])

    target_q = model.predict(states, verbose=0)
    next_q = model.predict(next_states, verbose=0)

    for i in range(batch_size):
        if dones[i]:
            target_q[i][actions[i]] = rewards[i]
        else:
            target_q[i][actions[i]] = rewards[i] + gamma * np.amax(next_q[i])

    model.fit(states, target_q, epochs=1, verbose=0)

# Training loop
episodes = 500

for e in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        replay()

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {e+1}/{episodes} - Reward: {total_reward} - Epsilon: {epsilon:.3f}")

env.close()
