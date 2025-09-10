import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Optional: parallel environments for faster training
# def make_env():
#     def _init():
#         env = gym.make("Ant-v4")
#         return Monitor(env)
#     return _init

# Use multiple environments in parallel (e.g., 4)
# env = SubprocVecEnv([make_env() for _ in range(4)])
env = gym.make("HalfCheetah-v4")
# Define PPO model
model = PPO("MlpPolicy", env, verbose=1, n_steps=1000, batch_size=64)

# Train the model
model.learn(total_timesteps=1_000_00)

# Save the model
model.save("ppo_ant")

# Clean up
env.close()
