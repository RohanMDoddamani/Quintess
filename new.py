
import gymnasium as gymn
import gym

env = gymn.make('Taxi-v3')
observation, info = env.reset()
print(env.observation_space)
print(env.action_space)
g = []
for i in range(10):
    a,b,c,d,e = env.step(env.action_space.sample())
    g.append(b)


print(g[::-1])
print(g)

def discountedReturn(gamma,y):
    t = 0
    for i in y:
        t = gamma*t + i
    return t

print(discountedReturn(0.3,g[::-1]))

