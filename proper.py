import gymnasium as gym
import imageio

env = gym.make("HalfCheetah-v4",render_mode="human")
obs, info =env.reset()

# g = env.render()
# print(g)

frames = []
for _ in range(100):
    frame = env.render()
    frames.append(frame)
    action = env.action_space.sample()  
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
imageio.mimsave('mujoco_run.mp4', frames, fps=30)
print("Video saved as mujoco_run.mp4")
