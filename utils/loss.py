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


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class DiceLossV2(nn.Module):
    def __init__(self, smooth=1., ignore_index=11):
        super(DiceLossV2, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):

        # target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        # print(target)
        # exit()
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class OhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_lb=11):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


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


class SoftLoss(nn.Module):
    def __init__(self):
        super(SoftLoss, self).__init__()

    def forward(self, output, target):
        assert output.shape == target.shape, "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        intersect = (output * target).sum(-1)
        loss = -torch.log(intersect)
        return torch.mean(loss)


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)
