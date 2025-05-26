import torch
import torch.nn as nn


class ObjectFeatureFusioner(nn.Module):
    def __init__(
            self,
            cl_dim = 128,
            cs_dim = 512
    ):
        super().__init__()
        self.fc = nn.Linear(cs_dim, cl_dim)

    def forward(self, feature_x, feature_y):
        b = feature_x.size(0)
        feature_y = feature_y.reshape(b, 1, feature_y.shape[1])
        feature_y = self.fc(feature_y)
        return feature_x + feature_y
