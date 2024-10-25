import torch


def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))


def dice_loss(y_real, y_pred):
    return 1 - (torch.mean(2 * torch.mul(y_real, y_pred) + 1) / (torch.mean(torch.add(y_real, y_pred)) + 1))