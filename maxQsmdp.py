import gym, numpy as np, random
from collections import defaultdict

env = gym.make("Taxi-v3")

# --- Task definitions ---
class Task:
    def __init__(self, name, actions, is_terminal):
        self.name = name
        self.actions = actions      # can be primitive env actions or other Tasks
        self.is_terminal = is_terminal

# Terminal conditions
def root_term(obs): return False  # root runs until success
def nav_term_factory(goal_idx):
    def term(obs):
        taxi_row, taxi_col, pass_loc, dest_idx = obs
        return pass_loc == goal_idx or pass_loc == 4
    return term


def decode_observation(encoded_obs):
    # Reverse the encoding formula to extract individual components
    # print(encoded_obs)
    dest_idx = encoded_obs % 4
    pass_loc = (encoded_obs // 4) % 5
    taxi_row = (encoded_obs // (4 * 5)) % 5
    taxi_col = (encoded_obs // 4) % 5
    return taxi_row, taxi_col, pass_loc, dest_idx
# Define primitive
primitive_actions = list(range(env.action_space.n))
primitive = Task("primitive", primitive_actions, lambda obs: True)

# Subtasks
go_pick    = Task("go_pick", primitive_actions, nav_term_factory(obs[2] for obs in []))
go_drop   = Task("go_drop", primitive_actions, nav_term_factory(obs[3] for obs in []))
pickup     = Task("pickup", primitive_actions, lambda obs: obs[2]!=4)
dropoff    = Task("dropoff", primitive_actions, lambda obs: obs[2]==4)

root = Task("root", [go_pick, pickup, go_drop, dropoff], lambda obs: obs[2]==4 and obs[3]==4)

# --- Value tables ---
V = defaultdict(float)
C = defaultdict(float)

epsilon, alpha, gamma = 0.1, 0.5, 0.99

def greedy(task, obs):
    if random.random() < epsilon:
        return random.choice(task.actions)
    vals = []
    for a in task.actions:
        vals.append(V[(a.name if isinstance(a, Task) else a, obs)])
    return task.actions[np.argmax(vals)]

def MAXQ(i: Task, obs):
    if isinstance(i, int):  # primitive action
        ns, r, done, _ = env.step(i)
        V[(i, obs)] += alpha * (r - V[(i, obs)])
        return ns, 1, r, done

    total_t = 0; total_r = 0; current_obs = obs
    while not i.is_terminal(current_obs):
        a = greedy(i, current_obs)
        if isinstance(a, Task):
            ns, t, r, done = MAXQ(a, current_obs)
        else:
            ns, r, done, _ = env.step(a)
            t = 1
            V[(a, current_obs)] += alpha * (r - V[(a, current_obs)])
        total_t += t
        total_r += r

        future = max(V[(b if isinstance(b, int) else b.name, ns)] for b in i.actions)
        C[(i.name, current_obs, a if isinstance(a, int) else a.name)] += alpha * ((gamma**t) * future - C[(i.name, current_obs, a if isinstance(a, int) else a.name)])
        current_obs = ns
        if done:
            break
    return current_obs, total_t, total_r, done

# --- Train ---
for ep in range(1):
    obs = env.reset()
    done = False
    while not done:
        obs, _, _, done = MAXQ(root, obs)
