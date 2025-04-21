import torch
import torch.nn as nn
import torch.nn.functional as F

class DualStreamNet(nn.Module):
    def __init__(self):
        super(DualStreamNet, self).__init__()
        
        # Goal & bot positions stream (4 values total)
        self.pos_fc1 = nn.Linear(4, 32)
        self.pos_fc2 = nn.Linear(32, 32)

        # Grid stream (9x9 = 81 values, treat it like a 1-channel image)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # (1, 9, 9) -> (8, 9, 9)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # -> (16, 9, 9)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # -> (32, 9, 9)
        self.grid_fc = nn.Linear(32 * 9 * 9, 64)  # Flattened conv output

        # Combine both streams
        self.combined_fc1 = nn.Linear(32 + 64, 64)
        self.combined_fc2 = nn.Linear(64, 32)

        # Outputs
        self.action_head = nn.Linear(32, 6)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch_size, 85)
        pos = x[:, :4]     # shape: (batch_size, 4)
        grid = x[:, 4:]    # shape: (batch_size, 81)

        # Position stream
        pos = F.relu(self.pos_fc1(pos))
        pos = F.relu(self.pos_fc2(pos))

        # Grid stream
        grid = grid.view(-1, 1, 9, 9)        # reshape to (batch_size, 1, 9, 9)
        grid = F.relu(self.conv1(grid))
        grid = F.relu(self.conv2(grid))
        grid = F.relu(self.conv3(grid))
        grid = grid.view(grid.size(0), -1)   # Flatten
        grid = F.relu(self.grid_fc(grid))
        grid = .5 * grid

        # Combine streams
        combined = torch.cat([pos, grid], dim=1)
        combined = F.relu(self.combined_fc1(combined))
        combined = F.relu(self.combined_fc2(combined))

        # Output heads
        action_logits = self.action_head(combined)
        state_value = self.value_head(combined)

        return action_logits, state_value