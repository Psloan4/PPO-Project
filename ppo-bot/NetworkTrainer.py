from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

class NetworkTrainer:
    def __init__(self, network, epsilon=.2, lr=2e-4, num_epochs=2, minimbatch_size=48, c1=.75, c2=.01):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.epsilon = epsilon
        self.num_epochs = num_epochs
        self.minibatch_size = minimbatch_size
        self.c1 = c1
        self.c2 = c2
    
    def train(self, rollout_data):

        self.network.train()
        logs = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
        }
        n = 0

        states, actions, old_log_probs, rewards, values, dones = self.parse_data(rollout_data)

        advantages, returns = self.compute_gae(rewards, values.detach(), dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        print(f"advantages: mean={advantages.mean():.2f}, std={advantages.std():.2f}")
        # plot_advantages_and_returns(advantages, returns)

        # print(f"advantages: mean={advantages.mean():.2f}, std={advantages.std():.2f}")
        # print(f"returns: mean={returns.mean():.2f}, std={returns.std():.2f}")

        data_size = states.shape[0]
        indices = np.arange(data_size)

        for _ in range(self.num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, data_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
            
                logits, values_pred = self.network(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                mb_new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                prob_ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)

                clip_adv = torch.clamp(prob_ratio, 1 - self.epsilon, 1 + self.epsilon) * mb_advantages
                policy_loss = -torch.min(prob_ratio * mb_advantages, clip_adv).mean()

                value_loss = F.mse_loss(values_pred.squeeze(-1), mb_returns)

                entropy_bonus = entropy.mean()

                # print(f"policy: {policy_loss.item():.2f}, value: {value_loss.item():.2f}, entropy: {entropy_bonus.item():.2f}")

                loss = policy_loss + self.c1*value_loss - self.c2*entropy_bonus

                if n%1000 == 0:
                    n = 0
                    logs["policy_loss"].append(policy_loss.item())
                    logs["value_loss"].append(value_loss.item())
                    logs["entropy"].append(entropy_bonus.item())
                    logs["total_loss"].append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return logs, rewards.mean().item()



        
    def parse_data(self, rollout_data):
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
                actions.append(timestep["action"])  # action index
                old_log_probs.append(timestep["log_prob"])
                rewards.append(timestep["reward"])
                values.append(timestep["value"])
                dones.append(timestep["done"])

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        old_log_probs.detach()

        return states, actions, old_log_probs, rewards, values, dones

    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """
        rewards: [T]
        values: [T]
        dones: [T]
        """
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else values[t]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns


def plot_advantages_and_returns(advantages, returns, title="Advantages and Returns"):
    plt.figure(figsize=(12, 6))
    
    plt.plot(advantages, label='Advantages', alpha=0.7)
    plt.plot(returns, label='Returns', alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
