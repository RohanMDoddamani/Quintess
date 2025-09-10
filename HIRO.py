import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Hyperparameters
c = 10
total_episodes = 1000
max_steps = 200
gamma = 0.99
lr = 1e-3

env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Policy network for worker (input: obs + goal)
def build_policy():
    model = tf.keras.Sequential([
        layers.Input(shape=(obs_dim*2,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(act_dim, activation='tanh')])
    return model

# Manager policy network (input: obs, output: goal vector)
def build_manager():
    model = tf.keras.Sequential([
        layers.Input(shape=(obs_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(obs_dim, activation='tanh')])  # goal space same as obs
    return model

worker_policy = build_policy()
manager_policy = build_manager()

worker_optimizer = tf.keras.optimizers.Adam(lr)
manager_optimizer = tf.keras.optimizers.Adam(lr)

def discount_rewards(r, gamma=0.99):
    discounted = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted[t] = running_add
    return discounted

for episode in range(total_episodes):
    obs = env.reset()
    episode_worker_states = []
    episode_worker_actions = []
    episode_worker_rewards = []
    episode_manager_states = []
    episode_manager_goals = []
    episode_manager_rewards = []

    goal = manager_policy(np.expand_dims(obs, 0)).numpy()[0]  # manager picks initial goal
    total_reward = 0

    for step in range(max_steps):
        # Worker policy input: obs + goal
        worker_input = np.concatenate([obs, goal])
        worker_input = np.expand_dims(worker_input, 0)
        action = worker_policy(worker_input).numpy()[0]

        next_obs, ext_reward, done, _ = env.step(action)
        # Intrinsic reward = negative distance to goal in state space
        intrinsic_reward = -np.linalg.norm((obs + goal) - next_obs)

        # Save worker experience
        episode_worker_states.append(worker_input)
        episode_worker_actions.append(action)
        episode_worker_rewards.append(intrinsic_reward)

        total_reward += ext_reward

        # Manager chooses new goal every c steps
        if (step + 1) % c == 0:
            episode_manager_states.append(np.expand_dims(obs, 0))
            episode_manager_goals.append(goal)
            episode_manager_rewards.append(total_reward)
            total_reward = 0
            goal = manager_policy(np.expand_dims(next_obs, 0)).numpy()[0]

        obs = next_obs
        if done:
            break

    # Compute discounted rewards for worker and manager
    discounted_worker_rewards = discount_rewards(episode_worker_rewards, gamma)
    discounted_manager_rewards = discount_rewards(episode_manager_rewards, gamma)

    # Update worker policy with REINFORCE
    with tf.GradientTape() as tape:
        loss = 0
        for s, a, r in zip(episode_worker_states, episode_worker_actions, discounted_worker_rewards):
            pi = worker_policy(s)
            log_prob = -tf.reduce_sum((pi - a)**2)  # simple gaussian log_prob proxy
            loss += -log_prob * r
    grads = tape.gradient(loss, worker_policy.trainable_variables)
    worker_optimizer.apply_gradients(zip(grads, worker_policy.trainable_variables))

    # Update manager policy similarly
    with tf.GradientTape() as tape:
        loss = 0
        for s, g, r in zip(episode_manager_states, episode_manager_goals, discounted_manager_rewards):
            pi = manager_policy(s)
            log_prob = -tf.reduce_sum((pi - g)**2)
            loss += -log_prob * r
    grads = tape.gradient(loss, manager_policy.trainable_variables)
    manager_optimizer.apply_gradients(zip(grads, manager_policy.trainable_variables))

    print(f"Episode {episode+1} completed")

print("Training done!")



#######
import gym
import numpy as np
from gym import spaces

class HIROPendulumWrapper(gym.Env):
    def __init__(self, goal_interval=10):
        self.env = gym.make("Pendulum-v1")
        self.goal_interval = goal_interval
        self.t = 0
        self.goal = np.zeros(2)  # target direction (cos, sin)

        low = np.concatenate([self.env.observation_space.low, [-1, -1]])
        high = np.concatenate([self.env.observation_space.high, [1, 1]])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = self.env.action_space

    def reset(self):
        self.t = 0
        obs = self.env.reset()
        self.goal = self.sample_goal(obs)
        return np.concatenate([obs, self.goal])

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.t += 1

        if self.t % self.goal_interval == 0:
            self.goal = self.sample_goal(obs)

        shaped_reward = -np.linalg.norm(obs[:2] - self.goal)  # goal-based reward
        obs_aug = np.concatenate([obs, self.goal])
        return obs_aug, shaped_reward, done, truncated, info

    def sample_goal(self, obs):
        # Sample random direction or use heuristic
        angle = np.random.uniform(-np.pi, np.pi)
        return np.array([np.cos(angle), np.sin(angle)])

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()


def high_level_policy(obs):
    # e.g., based on pole angle
    theta = np.arctan2(obs[1], obs[0])
    return np.array([1.0 if theta < 0 else -1.0], dtype=np.float32)



############### SKELETON. ###############
import numpy as np

env = ...  # your base env (e.g., Pendulum-v1)

goal_interval = 10  # high-level sets goal every 10 steps

# High-level policy: maps state -> goal
def high_level_policy(state):
    # Just returns a fixed goal or random vector as example
    return np.random.uniform(-1, 1, size=(2,))

# Low-level policy: maps (state + goal) -> action
def low_level_policy(state_goal):
    # Simple example: random action
    return env.action_space.sample()

num_episodes = 5

for ep in range(num_episodes):
    state = env.reset()
    done = False
    step = 0

    # Initialize goal
    goal = high_level_policy(state)

    while not done:
        # Create low-level observation by appending goal to state
        low_level_obs = np.concatenate([state, goal])

        # Low-level action
        action = low_level_policy(low_level_obs)

        next_state, reward, done, info = env.step(action)

        # Accumulate reward etc. (omitted)
        ## distance between current goal & state

        step += 1
        state = next_state

        # Every goal_interval steps, update goal from high-level policy
        if step % goal_interval == 0:
            goal = high_level_policy(state)

    print(f"Episode {ep} finished")
