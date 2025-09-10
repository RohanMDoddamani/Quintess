# In pomdp, the agent has information about the environment, but doesn't have the visibility 
# current state 
# ( It has information about env, But doesn't know where it is in the env)
# In these cases,
#   1) First step is to find where we are in the env (find the current state) in minimal steps
#   2) Only After discovering the exact location of env, agent should start learning
#   
# 


# Belief method of finding the agent exact state (Bayesian bupdate of Belief)
#   1) Classify the env states with similar observability 
#   2) Assign equal probability to all the states you could possibly in
#   3) take a action (random), then observe the new state (see the new observation)
#   4) After looking at the observation in new state, go back and update 
#       the probability of previous state


# observation -> partial information of the state. or
            # -> signal from sensors
            # 




# G0000
# 0WWW0
# 00W00
# 0WWW0
# 0S000




#  pomdp variants -> partial visibility (only its surroundings)
#                 -> zero visibility ( in darkness)



# ds to define env and actions

import random
import numpy as np

class GridWorld:
    def __init__(self, rows, cols, start, goal, walls=None):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.walls = walls if walls else []
        self.agent_pos = self.start
        self.actions = ['up', 'down', 'left', 'right']
        
    def reset(self):
        """Reset the agent's position to the start state."""
        a = np.random.choice([0,4])
        # self.agent_pos = self.start
        self.agent_pos = (4,0)
        return self.agent_pos
        
    def step(self, action):
        """Move the agent according to the action, return new state, reward, and done flag."""
        if action == 'up':
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 'down':
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 'left':
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == 'right':
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        
        # Check if new position is out of bounds or a wall
        if (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols and new_pos not in self.walls):
            self.agent_pos = new_pos

        # Reward structure (e.g., goal=+1, each step=-0.1, etc.)
        reward = -0.1
        done = False
        if self.agent_pos == self.goal:
            reward = 1
            done = True

        return self.agent_pos, reward, done
        # return reward,done
    
    def render(self):
        """Print the grid and the agent's position."""
        print("Environment:-")
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    print('A', end=" ")  # A for Agent
                elif (r, c) == self.goal:
                    print('G', end=" ")  # G for Goal
                elif (r, c) in self.walls:
                    print('#', end=" ")  # # for Wall
                else:
                    print('.', end=" ")  # . for empty space
            print()

    def observe(self):
        """Print only the 3x3 grid surrounding the agent's current position."""
        r, c = self.agent_pos
        print('Observation :-')
        for i in range(r - 1, r + 2):  # rows from one above to one below agent
            for j in range(c - 1, c + 2):  # cols from one left to one right
                if 0 <= i < self.rows and 0 <= j < self.cols:
                    if (i, j) == self.agent_pos:
                        print('A', end=" ")
                    elif (i, j) == self.goal:
                        print('G', end=" ")
                    elif (i, j) in self.walls:
                        print('#', end=" ")
                    else:
                        print('.', end=" ")
                else:
                    # print(' ', end=" ")  # outside the grid boundary, print empty space
                    print('', end=" ")  # outside the grid boundary, print empty space
            print()




# Example of usage
grid = GridWorld(rows=5, cols=5, start=(0, 0), goal=(10,10), walls=[(1,1),(1,2),(1,3),(2,2),(3,1),(3,2),(3,3)])
grid.reset()
# grid.render()
grid.render()
print('-----------------')
grid.observe()
print('-----------------')
# Reset environment
# grid.reset()

# # Take some steps



# state, reward, done = grid.step('up')

# print('-----------------')
# grid.render()
# print('-----------------')
# grid.observe()

# state, reward, done = grid.step('up')
# print('-----------------')
# grid.render()
# print('-----------------')
# grid.observe()

# state, reward, done = grid.step('left')

# print('-----------------')
# grid.render()
# print('-----------------')
# grid.observe()
