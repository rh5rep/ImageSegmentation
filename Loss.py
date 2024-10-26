import torch


def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))


def dice_loss(y_real, y_pred):
    return 1 - (torch.mean(2 * torch.mul(y_real, y_pred) + 1) / (torch.mean(torch.add(y_real, y_pred)) + 1))

def cross_entropy_loss(y_real, y_pred):
        y_pred = torch.sigmoid(y_pred)
        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(y_pred, y_real)
        
        return loss

def cross_entropy_weighted_loss(y_real, y_pred, pos_weight=1.0):
    y_pred = torch.sigmoid(y_pred)
    loss_fn = torch.nn.BCELoss(weight=(y_real * (pos_weight - 1) + 1))
    loss = loss_fn(y_pred, y_real)
    
    return loss

# Link explaining focal loss: https://medium.com/elucidate-ai/an-introduction-to-focal-loss-b49d18cb3ded
def focal_loss(y_real, y_pred, gamma_2=2.0):
    prob = torch.sigmoid(y_pred)
    pt = torch.where(y_real == 1, prob, 1 - prob)
    focal_loss = - (1 - pt) ** gamma_2 * torch.log(pt + 1e-8)
    
    return torch.mean(focal_loss)