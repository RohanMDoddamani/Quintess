import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False)  # deterministic

#           [root]
#          /  |   \
#    [to_5][to_10][to_goal]
#        \     |     /
#            [navigate]

class Task:
    def __init__(self, name, actions, term_fn):
        self.name = name
        self.actions = actions  # can be primitive or subtasks
        self.term_fn = term_fn

    def is_terminal(self, state):
        return self.term_fn(state)


# Primitive actions
primitive_actions = [0, 1, 2, 3]  # L, D, R, U

# Terminal conditions
is_at_5 = lambda s: s == 5
is_at_10 = lambda s: s == 10
is_at_goal = lambda s: s == 15

# Primitive navigation task
navigate = Task("navigate", primitive_actions, lambda s: False)  # primitive

# Subtasks
move_to_5 = Task("move_to_5", [navigate], is_at_5)
move_to_10 = Task("move_to_10", [navigate], is_at_10)
move_to_goal = Task("move_to_goal", [navigate], is_at_goal)

# Root task
root = Task("root", [move_to_5, move_to_10, move_to_goal], is_at_goal)

Q = {
    "navigate": {},
    "move_to_5": {},
    "move_to_10": {},
    "move_to_goal": {},
    "root": {}
}



# def maxq_q_learning(task, state, alpha, gamma, Q):
#     if task.term_fn(state):
#         return

#     if task.actions[0] in primitive_actions:  # primitive
#         action = choose_action(Q[task.name], state)
#         next_state, reward, _, _, _ = env.step(action)
#         Q[task.name][(state, action)] = Q[task.name].get((state, action), 0) + \
#             alpha * (reward - Q[task.name].get((state, action), 0))
#         return next_state

#     else:  # composite task
#         while not task.term_fn(state):
#             subtask = select_subtask(task, state, Q)
#             state = maxq_q_learning(subtask, state, alpha, gamma, Q)
#         return state
