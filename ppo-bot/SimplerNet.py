import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplerNet(nn.Module):
    def __init__(self, use_dropout=False):
        super(SimplerNet, self).__init__()
        self.use_dropout = use_dropout

        # Shared encoder
        self.fc1 = nn.Linear(4, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 32)

        # Policy head
        self.action_head = nn.Linear(32, 6)

        # Value head (deeper, separate path)
        self.value_fc1 = nn.Linear(32, 64)
        self.value_ln1 = nn.LayerNorm(64)
        self.value_fc2 = nn.Linear(64, 32)
        self.value_ln2 = nn.LayerNorm(32)
        self.value_out = nn.Linear(32, 1)

        if self.use_dropout:
            self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x[:, :4]  # Use only the first 4 features

        # Shared encoder
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.fc3(x))

        # Policy output
        action_logits = self.action_head(x)

        # Value path (separate MLP)
        v = F.relu(self.value_ln1(self.value_fc1(x)))
        if self.use_dropout:
            v = self.dropout(v)
        v = F.relu(self.value_ln2(self.value_fc2(v)))
        if self.use_dropout:
            v = self.dropout(v)
        state_value = self.value_out(v)

        return action_logits, state_value
