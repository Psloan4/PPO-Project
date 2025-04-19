import torch
import torch.optim as optim
import torch.nn.functional as F

class NetworkTrainer:
    def __init__(self, network, gamma=.99, tau=.95, epsilon=.2, lr=3e04, num_epochs=4):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.num_epochs = num_epochs
    
    def train(self, rollout_data):

        self.network.train()

        # Convert rollout_data into usable tensors for training
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        values = []
        dones = []
        
        # Collect data
        for episode_data in rollout_data:
            for timestep in episode_data:
                states.append(timestep["state"])
                actions.append(timestep["action_raw"])  # action index
                old_log_probs.append(timestep["log_prob"])
                rewards.append(timestep["reward"])
                values.append(timestep["value"])
                dones.append(timestep["done"])
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
    
        # Compute returns
        returns = []
        next_value = 0  # At the end of the rollout, there’s no next value, so it’s 0 (or use a bootstrapped estimate)
        
        for t in reversed(range(len(rewards))):
            # Compute the return (sum of discounted rewards + value for the next timestep)
            if dones[t] == 1:  # If done, we stop adding future rewards
                next_value = 0
            returns.insert(0, rewards[t] + self.gamma * next_value)
            next_value = values[t]  # Update next value for the next step

        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages = []
        next_gae = 0
        for t in reversed(range(len(rewards))):
            # Compute delta (reward + discount * next_value - value) and use it for GAE
            delta = rewards[t] + self.gamma * next_value - values[t]
            next_gae = delta + self.gamma * self.tau * next_gae
            advantages.insert(0, next_gae)
            next_value = values[t]  # Update next value for the next step
        
        # Convert returns and advantages to tensors
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for i in range(self.num_epochs):
            action_logits, state_values = self.network(states)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = action_dist.log_prob(actions)

            ratios = torch.exp(log_probs - old_log_probs)

            clip_advantages = torch.clamp(ratios, 1 - self.epsilon, 1+self.epsilon) * advantages
            policy_loss = -torch.min(ratios * advantages, clip_advantages).mean()

            value_loss = F.mse_loss(state_values.squeeze(), returns)

            total_loss = policy_loss + .5 * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
