"""
Characteristic Line Encoder and Characteristic Line Decoder
We use nn.Conv1d instead of nn.Linear to extract the feature just like PointNet does.
Refer to: https://github.com/fxia22/pointnet.pytorch
"""
import torch
import torch.nn as nn


class CharacteristicLineEncoder(nn.Module):
    def __init__(
            self,
            dim = 3,
            input_length = 300,
            seq_length = 60,
            emb_dim = 128
    ):
        super().__init__()
        self.dim = dim
        self.input_length = input_length
        self.seq_length = seq_length
        self.emb_dim = emb_dim
        self._make_module()

    def _make_module(self):
        self.local_emb_nn = nn.Conv1d(self.dim, self.emb_dim, 1)
        num_points_per_local_regions = self.input_length // self.seq_length
        self.local_region_nn = nn.ModuleList([
            nn.Conv1d(num_points_per_local_regions, 1, 1)
            for _ in range(self.seq_length)
        ])
        self.global_emb_nn = nn.Conv1d(self.dim, self.emb_dim, 1)
        self.global_nn = nn.Conv1d(self.input_length, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        global_feature = self.relu(self.global_emb_nn(x)).transpose(1, 2)
        global_feature = self.relu(self.global_nn(global_feature))
        local_feature = self.relu(self.local_emb_nn(x)).transpose(1, 2)
        local_feature = list(torch.chunk(local_feature, self.seq_length, dim = 1))
        for i in range(self.seq_length):
            local_feature[i] = self.local_region_nn[i](local_feature[i])
        local_feature = torch.cat(local_feature, dim = 1)
        return local_feature + global_feature


class CharacteristicLineDecoder(nn.Module):
    def __init__(
            self,
            dim = 3,
            seq_length = 60,
            emb_dim = 128,
            output_length = 300
    ):
        super().__init__()
        self.dim = dim
        self.output_length = output_length
        self.seq_length = seq_length
        self.emb_dim = emb_dim
        self._make_module()

    def _make_module(self):
        self.re_emb_nn = nn.Conv1d(self.emb_dim, self.dim, 1)
        num_points_per_local_regions = self.output_length // self.seq_length
        self.re_local_region_nn = nn.ModuleList([
            nn.Conv1d(1, num_points_per_local_regions, 1)
            for _ in range(self.seq_length)
        ])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = list(torch.chunk(x, self.seq_length, dim = 1))
        for i in range(self.seq_length):
            x[i] = self.relu(self.re_local_region_nn[i](x[i])).transpose(1, 2)
        x = torch.cat(x, dim = 2)
        x = self.sigmoid(self.re_emb_nn(x))
        return x
