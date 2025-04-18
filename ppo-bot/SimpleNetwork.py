import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(85, 128)    # First hidden layer
        self.fc2 = nn.Linear(128, 64)    # Second hidden layer
        self.fc3 = nn.Linear(64, 32)     # Optional third hidden layer
        self.output = nn.Linear(32, 2)   # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))   # Activation after first layer
        x = F.relu(self.fc2(x))   # Activation after second layer
        x = F.relu(self.fc3(x))   # Activation after third layer
        return torch.sigmoid(self.output(x)) 