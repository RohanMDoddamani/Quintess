import gymnasium as gym
import random
from collections import defaultdict
import numpy as np

# Initialize environment
env = gym.make("FrozenLake-v1", is_slippery=False)
primitive_actions = [0, 1, 2, 3]  # Left, Down, Right, Up

# Define the Task class
class Task:
    def __init__(self, name, actions, term_fn):
        self.name = name
        self.actions = actions
        self.term_fn = term_fn

    def is_terminal(self, state):
        return self.term_fn(state)

# Terminal conditions
is_at_5 = lambda s: s == 5
is_at_10 = lambda s: s == 10
is_at_goal = lambda s: s == 15

# Primitive and composite tasks
navigate = Task("navigate", primitive_actions, is_at_10 or is_at_5 or is_at_goal)
move_to_goal = Task("move_to_10", [navigate], is_at_goal)
# move_to_5 = Task("move_to_5", [move_to_10], is_at_5)
move_to_10 = Task("move_to_10", [move_to_goal], is_at_10)
move_to_5 = Task("move_to_5", [move_to_10], is_at_5)
# move_to_goal = Task("move_to_goal", [navigate], is_at_goal)
# move_to_10 = Task("move_to_10", [move_to_goal], is_at_10)
root = Task("root", [move_to_5, move_to_10, move_to_goal], is_at_goal)
# root = Task("root", [move_to_5], is_at_goal)

# Q-table initialization
# Q = defaultdict(lambda: 0)
# H = { (i,j) :0 for i in range(16) for j in range(4)}
# print(H)

Q = {
    
    'navigate': { 
        (i,j) :0.0001 for i in range(16) for j in range(4)
        # (i,j) for i in range(16) for j in range(4)
        # (0, 1): 0.3,  # From state 0, taking action 1 (Down) has Q = 0.3
        # (4, 1): 0.4,  # From state 4, Down = 0.4
        # (5, 2): 0.6,  # From state 5, Right = 0.6
     
    },
    'move_to_5': {
        # (0, navigate): 0.5,
        # (1, navigate): 0.3,
         (i,navigate.name) :0.0001 for i in range(16)

    },
    'move_to_10': {
        # (5, navigate): 0.6,
        # (6, navigate): 0.000001.4,
        (i,navigate.name) :0.0001 for i in range(16)
 
    },
    'move_to_goal': {
        (i,navigate.name) :0.0001 for i in range(16)
        # (10, navigate): 1.0,
        # (11, navigate): 0.8,
     
    },
    'root': {
        # (0, move_to_5): 0.5,
        # (5, move_to_10): 0.6,
        # (10, move_to_goal): 1.0,
     (j,i.name) :0.0001 for j in range(16) for i in [move_to_5, move_to_10, move_to_goal]
     
    }
}
# Q = {**Q,**H}
print(Q['move_to_10'])


# Cost-to-go table: C[parent_task][(state, subtask)] = expected cost after subtask
C = Q
print("LEKJF")
# print(Q)
# print(C['navigate'])

# print(Q['move_to_5'])

# Helper functions
def choose_action(Q_table, state, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(primitive_actions)
    q_values =[Q['navigate'][(state, a)] for a in primitive_actions]
    # print(q_values)
    # q_values = [Q_table.get((state, a), 0.0) for a in primitive_actions]
    max_q = max(q_values)
    best_action = [a for a in range(4) if Q['navigate'][(state,a)] == max_q]
    # print('best',best_action)
    return np.random.choice(best_action)

def select_subtask(task, state, Q):
    # For simplicity, execute subtasks in order
    for subtask in task.actions:
        if not subtask.is_terminal(state):
            return subtask
    # return None

# MAXQ recursive learning function
# def maxq_q_learning(task, state, alpha, gamma, Q, total_reward=0):
#     if task.is_terminal(state):
#         return state, total_reward

#     if task.name == "navigate":
#         action = choose_action(Q[task.name], state)
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         Q[task.name][(state, action)] += alpha * (reward - Q[task.name][(state, action)])
#         return next_state, total_reward + reward

#     else:
#         while not task.is_terminal(state):
#             subtask = select_subtask(task, state, Q)
#             if subtask is None:
#                 break
#             state, total_reward = maxq_q_learning(subtask, state, alpha, gamma, Q, total_reward)
#         return state, total_reward


# def train_maxq(env, root_task, episodes=10, alpha=0.1, gamma=0.99):
#     for episode in range(episodes):
#         state, _ = env.reset()
#         maxq_q_learning(root_task, state, alpha, gamma, Q)
#     return Q


def maxq_q_learning(task, state, alpha, gamma, Q, C, total_reward=0):
    if task.is_terminal(state):
        return state, total_reward

    if task.name == "navigate" :
        for i in range(10):
            action = choose_action(Q[task.name], state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            Q[task.name][(state, action)] += alpha * (reward - Q[task.name][(state, action)])
            return next_state, total_reward + reward

    # else:
    #     while not task.is_terminal(state):
    #         subtask = select_subtask(task, state, Q)
    #         if subtask is None:
    #             break
    #         state, total_reward = maxq_q_learning(subtask, state, alpha, gamma, Q, total_reward)
    #     return state, total_reward
    else:
        # while not task.is_terminal(state):
        for i in range(10) :
            while not task.is_terminal(state):
                subtask = select_subtask(task, state, Q)
                # print(subtask.name)
                print(subtask.name) if subtask.name is not 'navigate' else None
          
                if subtask is None:
                    break

                sub_start_state = state
                state, sub_reward = maxq_q_learning(subtask, state, alpha, gamma, Q, C)

          
                # Estimate cost-to-go from state after subtask
                future_cost = 0.0
                for next_sub in task.actions:
                    if not next_sub.is_terminal(state):
                  
                        future_cost = C[task.name][(state, next_sub.name)]
                        break

          
            # print([task.name])
                current_cost = C[task.name][(sub_start_state, subtask.name)]
                target_cost = sub_reward + gamma * future_cost
                C[task.name][(sub_start_state, subtask.name)] += alpha * (target_cost - current_cost)

                total_reward += sub_reward

        return state, total_reward

def train_maxq(env, root_task, episodes=1, alpha=0.1, gamma=0.7):
    for episode in range(1):
        state, _ = env.reset()
        maxq_q_learning(root_task, state, alpha, gamma, Q, C)
        # print(rr)
    return Q, C


# Q_learned = train_maxq(env, root)
Q_learned, C_learned = train_maxq(env, root)

# print(Q_learned[)
