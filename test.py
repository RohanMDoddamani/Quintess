import numpy as np, random
import gymnasium as gym
from collections import defaultdict

env = gym.make("Taxi-v3")

# --- Task definitions ---
class Task:
    def __init__(self, name, actions, is_terminal):
        self.name = name
        self.actions = actions      # can be primitive env actions or other Tasks
        self.is_terminal = is_terminal


s = env.reset()  # Get initial observation
# print(s[0])

def decode_observation(encoded_obs):
    # Reverse the encoding formula to extract individual components
    # print(encoded_obs)
    dest_idx = encoded_obs % 4
    pass_loc = (encoded_obs // 4) % 5
    taxi_row = (encoded_obs // (4 * 5)) % 5
    taxi_col = (encoded_obs // 4) % 5
    return taxi_row, taxi_col, pass_loc, dest_idx

obs = decode_observation(s[0])
goal_idx = obs[2]
# print(obs)
# print(goal_idx)

primitive_actions = list(range(env.action_space.n))

def nav_term_factory(goal_idx):
    def term(obs):
        taxi_row, taxi_col, pass_loc, dest_idx = decode_observation(obs)
        return pass_loc == goal_idx or pass_loc == 4
    return term
def ggg(obs):
    print('gggg',obs)
    # print(decode_observation(obs))
    return obs[2] == 4

go_pick = Task("go_pick", primitive_actions, lambda obs: obs[2]!=4)
go_drop = Task("go_drop", primitive_actions, lambda obs: nav_term_factory(obs[3]))
pickup  = Task("pickup", primitive_actions, lambda obs: obs[2]!=4)
dropoff = Task("dropoff", primitive_actions, lambda obs: ggg(obs))

# print()

root = Task("root", [go_pick, pickup, go_drop, dropoff], lambda obs: obs[2]==4 and obs[3]==4)

print(go_drop.is_terminal((0,0,4,0)))

V = defaultdict(float)
C = defaultdict(float)

epsilon = 0.3

def greedy(task, obs):
    # print(obs)
    if random.random() < epsilon:
        return random.choice(task.actions)
    vals = []
    for a in task.actions:
        vals.append(V[(a.name if isinstance(a, Task) else a, obs)])
    # print(task.actions[np.argmax(vals)])
    return task.actions[np.argmax(vals)]


epsilon, alpha, gamma = 0.1, 0.5, 0.99

# print(root.is_terminal(obs))
# print(go_drop.is_terminal(obs))

# def MAXQ(i: Task, obs):
#     if isinstance(i, int):  # primitive action
#         ns, r, done, _ ,info= env.step(i)
 
#         disk = decode_observation(obs[0])
#         V[(i, disk)] += alpha * (r - V[(i, disk)])
#         print(disk)
#         return ns, 1, r, done
    
#     total_t = 0; total_r = 0; current_obs = decode_observation(obs[0])
#     print(decode_observation(obs[0]))

#     done =False
#     # print(i,i.is_terminal(decode_observation(current_obs[0])))
    
#     while not i.is_terminal(decode_observation(obs[0])):
#         current_obs = decode_observation(obs[0])
#         a = greedy(i, current_obs)
#         print(a)
#         # break
#         if isinstance(a, Task):
#             ns, t, r, done = MAXQ(a, obs)
#             # print(a)
            

#         else:
#             ns, r, done, _ ,info= env.step(a)
#             t = 1
#             V[(a, current_obs)] += alpha * (r - V[(a, current_obs)])
#         total_t += t
#         total_r += r
#         future = max(V[(b if isinstance(b, int) else b.name, ns)] for b in i.actions)


#         C[(i.name, current_obs, a if isinstance(a, int) else a.name)] += alpha * ((gamma**t) * future - C[(i.name, current_obs, a if isinstance(a, int) else a.name)])
#         current_obs = ns

#         if done:
#             break
    
#     return obs, total_t, total_r, done



# print('hee',decode_observation(226))
# --- Train ---

def MAXQ(i: Task, obs):
    print('m',obs)
    if isinstance(i, int):  # primitive action
        # print(i)
        ns, r, done, _ ,info= env.step(i)
        # dick = decode_observation(obs[0])
        # dick = obs
        V[(i, obs)] += alpha * (r - V[(i, obs)])
        # print(obs)
        # print('t',decode_observation(ns))
        return decode_observation(ns), 1, r, done

    total_t = 0; total_r = 0; current_obs = obs
    done =False
    # print('top',current_obs[2])
    while not i.is_terminal(current_obs):
        a = greedy(i, current_obs)
        # print('helsjflk',current_obs)
        # break
        if isinstance(a, Task):

            decode_observation(ns), t, r, done = MAXQ(a, current_obs)
            # print('ggg',ns)
        else:
            ns, r, done, _,info = env.step(a)
            t = 1
            # print('elese',current_obs)
            # print('en',ns)
            V[(a, current_obs)] += alpha * (r - V[(a, current_obs)])
        total_t += t
        total_r += r

        # print('this', ns)
        # print('curr',current_obs)
        future = max(V[(b if isinstance(b, int) else b.name, ns)] for b in i.actions)
        C[(i.name, current_obs, a if isinstance(a, int) else a.name)] += alpha * ((gamma**t) * future - C[(i.name, current_obs, a if isinstance(a, int) else a.name)])
        current_obs = ns
        if done:
            break
        # print('g',decode_observation(current_obs))
    return current_obs, total_t, total_r, done

for ep in range(1):
    obs = env.reset()
    done = False
    obs = decode_observation(obs[0])
    # print(obs)
    for i in range(1):
    # # while not done:
        decode_observation(obs), _, _, done = MAXQ(root,obs)
        # print(obs)
    
# print(C)



