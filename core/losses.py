import torch
import torch.nn.functional as f


def loss_p(strip, pred_strip, coordinate_weights):
    loss = 0
    for i in range(3):
        loss += torch.mean(
            torch.norm(strip[:, i] - pred_strip[:, i], 2, 1)
        ) * coordinate_weights[i]
    return loss


def loss_r(section, recover_section):
    return f.mse_loss(section, recover_section)
