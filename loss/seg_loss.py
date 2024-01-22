import numpy as np
import torch
from torch import nn

__all__ = ["SoftDiceLoss", "IoULoss"]

"""
Codes are referenced from https://github.com/JunMa11/SegLossOdyssey.
"""

def get_dice(pred, target, smooth = 1.):
    '''dice = (2 * intersection(pred, target)) / (pred + target)'''
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., channel_weights = [1.]):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf
        """
        super(SoftDiceLoss, self).__init__()
        
        self.smooth = smooth
        self.channel_weights = channel_weights
    
    def forward(self, pred, target):
        b, c, h, w = pred.shape
        if len(self.channel_weights) != c:
            self.channel_weights = np.ones(c)*self.channel_weights[0]
        loss = 0
        for i in range(c):
            loss += (1 - get_dice(pred[:, i, :, :], target[:, i, :, :], smooth = self.smooth)) * self.channel_weights[i]
        return loss

def get_iou(pred, target):
    '''intersection of union = tp / fp + tp + fn'''
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    return tp / (fp + tp + fn)

class IoULoss(nn.Module):
    def __init__(self, 
                 smooth=None, channel_weights = None):
        """
        paper: https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22
        """
        super(IoULoss, self).__init__()

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        assert len(self.channel_weights) == c
        loss = 0
        for i in range(c):
            loss += (1 - get_iou(pred[:, i, :, :], target[:, i, :, :])) * self.channel_weights[i]
        return loss

class FocalLoss():
    def __init__(self):
        pass 