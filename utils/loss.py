import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from texar.torch.losses import entropy

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.shape == target.shape, "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator

#
# class EntropyLoss(nn.Module):
#     def __init__(self):
#         super(EntropyLoss, self).__init__()
#
#     def forward(self, v):
#         """
#             Entropy loss for probabilistic prediction vectors
#             input: batch_size x channels x h x w
#             output: batch_size x 1 x h x w
#         """
#         assert v.dim() == 4
#         n, c, h, w = v.size()
#         print(np.log2(c))
#         print(-torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c)))
#         print(-torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c)))
#         exit()
#         return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))
#


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    r"""Entropy of prediction.
    The definition is:
    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c
    where C is number of classes.
    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``
    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """

    def forward(self, predictions):
        predictions = F.softmax(predictions, dim=1)
        epsilon = 1e-5
        h = -predictions * torch.log(predictions + epsilon)
        h = h.sum(dim=1).mean()
        return h