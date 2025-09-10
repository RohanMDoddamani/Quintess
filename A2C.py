import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


env = gym.make("CartPole-v1")

# Hyperparameters
learning_rate = 1e-2 #0.001
gamma = 0.6

# Actor-Critic Network
class A2CNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x)

# Initialize
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = A2CNet(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for episode in range(300):
    state = env.reset()[0]
    log_probs = []
    values = []
    rewards = []

    done = False
    while not done:
        state_tensor = torch.FloatTensor(state)
        logits, value = model(state_tensor)
        probs = torch.softmax(logits.unsqueeze(0),dim=1)
        dist = torch.distributions.Categorical(probs[0])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        next_state, reward, done, _,one = env.step(action.item())         

        log_probs.append(dist.log_prob(action))
        values.append(value)
        rewards.append(reward)

        state = next_state

    # Compute returns and advantages
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    values = torch.cat(values)
    log_probs = torch.stack(log_probs)

    advantages = returns - values.squeeze()

    # Compute losses
    actor_loss = -(log_probs * advantages.detach()).mean()
    critic_loss = advantages.pow(2).mean()
    loss = actor_loss + critic_loss

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    if episode % 5 == 0:
        print(f"Episode {episode}, Total reward: {sum(rewards)}")

env.close()





