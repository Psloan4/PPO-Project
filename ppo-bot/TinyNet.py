import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.fc = nn.Linear(4, 32)
        self.action_head = nn.Linear(32, 6)
        self.value_head  = nn.Linear(32, 1)

    def forward(self, x):
        x = x[:, :4]              # ← take only the first 4 dims
        x = F.relu(self.fc(x))
        action_logits = self.action_head(x)
        state_value   = self.value_head(x)
        return action_logits, state_value
