import gymnasium as gym
from stable_baselines3 import PPO
e = gym.make("HalfCheetah-v4", render_mode="human")
model = PPO.load("ppo_ant")

obs, info = e.reset()
done = False
r = 0

print(e.observation_space)
print(e.action_space)

while not done:
# for i in range(2):
    action, _ = model.predict(obs)
    # print(action)
    obs, reward, terminated, truncated, info = e.step(action)
    
    print(reward)
    r += reward
    done = terminated or truncated
    e.render()
for j in range(300):
    e.render()

# print(reward)
e.close()