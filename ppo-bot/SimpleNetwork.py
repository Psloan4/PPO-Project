import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(85, 128)    # First hidden layer
        self.fc2 = nn.Linear(128, 64)    # Second hidden layer
        self.fc3 = nn.Linear(64, 32)     # Optional third hidden layer
        
        # A single head that outputs 6 logits for the 6 possible actions
        self.action_head = nn.Linear(32, 6)  # 6 possible actions (left_jump, left_stay, left_no_jump, right_jump, right_stay, right_no_jump)
        self.value_head = nn.Linear(32, 1)  # Single scalar value for the state value

    def forward(self, x):
        x = F.relu(self.fc1(x))   # Activation after first layer
        x = F.relu(self.fc2(x))   # Activation after second layer
        x = F.relu(self.fc3(x))   # Activation after third layer

        action_logits = self.action_head(x)
        state_value = self.value_head(x)
        return action_logits, state_value