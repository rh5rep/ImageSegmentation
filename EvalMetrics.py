import torch


def dice_overlap(y_real, y_pred):
    intersection = torch.sum(torch.mul(y_real, y_pred))
    denominator = torch.sum(y_real) + torch.sum(y_pred)

    return (2 * intersection) / denominator


def intersection_over_union(y_real, y_pred):
    intersection = torch.sum(torch.mul(y_real, y_pred))
    union = torch.sum(y_real) + torch.sum(y_pred) - intersection

    return intersection/union


def accuracy(y_real, y_pred):
    y_real = y_real.view(-1)
    y_pred = y_pred.view(-1)
    correct_predictions = torch.sum(y_real == y_pred)
    total_pixels = y_real.size(0)

    return correct_predictions.float() / total_pixels


def sensitivity(y_real, y_pred):
    y_real = y_real.view(-1)
    y_pred = y_pred.view(-1)
    tp = torch.sum((y_real == 1) & (y_pred == 1))
    fn = torch.sum((y_real == 1) & (y_pred == 0))
    
    return tp / (tp + fn)


def specificity(y_real, y_pred):
    tn = torch.sum((y_real == 0) & (y_pred == 0))
    fp = torch.sum((y_real == 0) & (y_pred == 1)) 

    return tn / (tn + fp)
