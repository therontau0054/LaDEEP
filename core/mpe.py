import torch
import torch.nn as nn


class MotionParametersEncoder(nn.Module):
    def __init__(
            self,
            seq_length = 60,
            emb_dim = 128,
            motion_degrees = 6
    ):
        super().__init__()
        self.degrees = motion_degrees
        seq_for_per_degree = seq_length // motion_degrees
        self.fc = nn.ModuleList([
            nn.Conv1d(1, seq_for_per_degree, 1)
            for _ in range(motion_degrees)
        ])
        self.emb_nn = nn.Linear(1, emb_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.emb_nn(x.transpose(1, 2)))
        x = list(torch.chunk(x, self.degrees, dim = 1))
        for i in range(self.degrees):
            x[i] = self.relu(self.fc[i](x[i]))
        x = torch.cat(x, dim = 1)
        return x
