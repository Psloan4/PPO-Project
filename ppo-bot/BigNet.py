import torch
import torch.nn as nn
import torch.nn.functional as F

class BigNet(nn.Module):
    def __init__(self):
        super(BigNet, self).__init__()
        self.fc1 = nn.Linear(85, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)

        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(256)
        self.ln4 = nn.LayerNorm(128)
        self.ln5 = nn.LayerNorm(64)

        # Output heads
        self.action_head = nn.Linear(64, 6)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        x = F.relu(self.ln4(self.fc4(x)))
        x = F.relu(self.ln5(self.fc5(x)))

        action_logits = self.action_head(x)
        state_value = self.value_head(x)
        return action_logits, state_value
