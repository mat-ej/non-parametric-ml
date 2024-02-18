import torch 
from torch import nn
from torch.nn import functional
import torch.nn.functional as F

class PinballLoss(nn.Module):
    """
    Calculates the quantile loss function.

    Attributes
    ----------
    self.quantiles : torch.tensor
    """
    def __init__(self, quantiles):
        super(PinballLoss, self).__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float))
        
    def forward(self, pred, target):
        error = target - pred
        upper =  self.quantiles * error
        lower = (self.quantiles - 1) * error 

        losses = torch.max(lower, upper)
        loss = torch.mean(torch.sum(losses, dim=1))
        return loss

class SmoothPinballLoss(nn.Module):
    """
    Smoth version of the pinball loss function.

    Parameters
    ----------
    quantiles : torch.tensor
    alpha : int
        Smoothing rate.

    Attributes
    ----------
    self.quantiles : torch.tensor
    """
    def __init__(self, quantiles, alpha=0.01):
        super(SmoothPinballLoss,self).__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float))
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        """
        error = target - pred
        q_error = self.quantiles * error
        beta = 1 / self.alpha
        soft_error = F.softplus(-error, beta)

        losses = q_error + soft_error
        loss = torch.mean(torch.sum(losses, dim=1))
        return loss