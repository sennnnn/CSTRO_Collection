import torch
import torch.nn as nn
import torch.nn.functional as F

class Cross_Entropy_Loss(object):
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
    
    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        target = target.long()

        loss_value = self.criterion(logit, target)

        return loss_value


class Custom_Cross_Entropy_Loss(object):
    def __init__(self, weight=None):
        self.weight = weight

    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        target = F.one_hot(target.long(), c).permute(0, 3, 1, 2)
        logit = F.softmax(logit, dim=1)

        cross_entropy = -1*target*torch.log(logit)

        loss_value = torch.mean(torch.sum(Cross_Entropy, dim=1))

        return loss_value


class Focal_Loss(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        target = F.one_hot(target.long(), c).permute(0, 3, 1, 2)
        logit = F.softmax(logit, dim=1)
        
        focal_entropy = -1*torch.pow(1-logit, self.alpha)*target*torch.log(logit)

        loss_value = torch.mean(torch.sum(focal_entropy, dim=1))

        return loss_value


class Dice_Loss(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        target = F.one_hot(target.long(), c).permute(0, 3, 1, 2)
        logit = F.softmax(logit, dim=1)

        A_vector = torch.sum(logit, dim=(0, 2, 3))
        B_vector = torch.sum(target, dim=(0, 2, 3))
        AB_vector = torch.sum(logit*target, dim=(0, 2, 3))
        dice_value = (AB_vector + self.epsilon) / (A_vector + B_vector + self.epsilon)

        loss_value = c - torch.sum(dice_value, dim=0)

        return loss_value


class Log_Cosh_Dice_Loss(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        target = F.one_hot(target.long(), c).permute(0, 3, 1, 2)
        logit = F.softmax(logit, dim=1)

        A_vector = torch.sum(logit, dim=(0, 2, 3))
        B_vector = torch.sum(target, dim=(0, 2, 3))
        AB_vector = torch.sum(logit*target, dim=(0, 2, 3))
        dice_value = (AB_vector + self.epsilon) / (A_vector + B_vector + self.epsilon)

        loss_value = torch.log(torch.cosh(c - torch.sum(dice_value, dim=0)))

        return loss_value


class Hausdorff_Distance_Loss(object):
    def __init__(self):
        pass


def construct_loss(params):
    which = params["LOSS_TYPE"]
    if which == "CROSS_ENTROPY":
        return Cross_Entropy_Loss()
    elif which == "FOCAL":
        return Focal_Loss(params["FOCAL_ALPHA"])
    elif which == "DICE":
        return Dice_Loss(params["DICE_EPSILON"])
    elif which == "LOG_COSH_DICE":
        return Log_Cosh_Dice_Loss(params["DICE_EPSILON"])

