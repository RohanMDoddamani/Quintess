import pomdp_py
import numpy as np

# Define States
class TigerState(pomdp_py.State):
    def __init__(self, tiger_pos):
        self.tiger_pos = tiger_pos  # "left" or "right"
    def __hash__(self):
        return hash(self.tiger_pos)
    def __eq__(self, other):
        return self.tiger_pos == other.tiger_pos
    def __repr__(self):
        return f"TigerState({self.tiger_pos})"

# Define Actions
class Listen(pomdp_py.Action):
    def __init__(self):
        super().__init__("listen")
class OpenLeft(pomdp_py.Action):
    def __init__(self):
        super().__init__("open_left")
class OpenRight(pomdp_py.Action):
    def __init__(self):
        super().__init__("open_right")

# Define Observations
class HearLeft(pomdp_py.Observation):
    def __init__(self):
        super().__init__("hear_left")
class HearRight(pomdp_py.Observation):
    def __init__(self):
        super().__init__("hear_right")

# Transition Model (Tiger position never changes)
class TigerTransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        return 1.0 if next_state == state else 0.0
    def sample(self, state, action):
        return state

# Observation Model
class TigerObservationModel(pomdp_py.ObservationModel):
    def probability(self, observation, next_state, action):
        if isinstance(action, Listen):
            if next_state.tiger_pos == "left":
                return 0.85 if isinstance(observation, HearLeft) else 0.15
            else:
                return 0.85 if isinstance(observation, HearRight) else 0.15
        else:
            # No informative observation on open actions
            return 0.5
    def sample(self, next_state, action):
        import random
        if isinstance(action, Listen):
            if next_state.tiger_pos == "left":
                return HearLeft() if random.random() < 0.85 else HearRight()
            else:
                return HearRight() if random.random() < 0.85 else HearLeft()
        else:
            # Random observation when opening
            return HearLeft() if random.random() < 0.5 else HearRight()

# Reward Model
class TigerRewardModel(pomdp_py.RewardModel):
    def sample(self, state, action, next_state):
        if isinstance(action, Listen):
            return -1
        elif isinstance(action, OpenLeft):
            return 10 if state.tiger_pos == "right" else -100
        elif isinstance(action, OpenRight):
            return 10 if state.tiger_pos == "left" else -100
        return 0

# Define Policy Model (dummy, we do VI manually)
class TigerPolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self):
        super().__init__()
    def sample(self, state):
        return Listen()

# Create POMDP Model
transition_model = TigerTransitionModel()
observation_model = TigerObservationModel()
reward_model = TigerRewardModel()
policy_model = TigerPolicyModel()

# model = pomdp_py.POMCP(transition_model, observation_model, reward_model, policy_model)

# Initial belief: Uniform over tiger left/right
belief = pomdp_py.Histogram({TigerState("left"): 0.5, TigerState("right"): 0.5})

# Now, we can use pomdp_py's ValueIteration solver
vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
# vi.set_model(model)
# vi.set_belief(belief)

# vi.solve()  # Runs value iteration

print("Computed value function:")
for alpha in vi.alpha_vectors:
    print(alpha)
    
# You can query the best action for current belief:
best_action = vi.get_action(belief)
print(f"Best action for belief {belief} is {best_action}")
