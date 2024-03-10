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
    needs_logits: bool = False

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
    needs_logits: bool = False

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

class BinarizedQuantileLoss(nn.Module):

    needs_logits: bool = True

    def __init__(self, quantiles):
        super(BinarizedQuantileLoss, self).__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float))
        self.qloss_low = PinballLoss(quantiles[0])
        self.qloss_high = PinballLoss(quantiles[-1])

    @staticmethod
    def find_bin_indices(y_true, q_low, q_high, n_bins):
        range = q_high - q_low
        bin_size = range / n_bins
        bin_indices = ((y_true - q_low) / bin_size).floor().long()
        bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)
        return bin_indices
        
    def forward(self, q_hat, y_true, logits):
        q_low = q_hat[:, 0:1]
        q_high = q_hat[:, -1:]
        q_interm = logits  # Use logits directly for cross-entropy

        bin_labels = self.find_bin_indices(y_true, q_low, q_high, q_interm.size(1)).squeeze()
        ce_loss = F.cross_entropy(logits, bin_labels)
        q_loss = self.qloss_low(q_low, y_true) + self.qloss_high(q_high, y_true)

        return ce_loss + q_loss