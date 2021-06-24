import numpy as np
import torch
from torch.cuda.amp import autocast

from utils import reverse_one_hot, global_accuracy, get_confusion_matrix, intersection_over_union


def train_segmentation(model, data, label, optimizer, scaler, criterion, loss):
    # Set model to Train mode
    model.train()

    # Clear optimizer gradient in an efficient way
    optimizer.zero_grad(set_to_none=True)

    with autocast():
        # Get network output
        output, output_sup1, output_sup2 = model(data)

        # Loss
        loss1 = criterion(output, label)
        loss2 = criterion(output_sup1, label)
        loss3 = criterion(output_sup2, label)
        loss = loss1 + loss2 + loss3

    # Compute gradients with gradient scaler
    scaler.scale(loss).backward()

    scaler.step(optimizer)
    # Updates the scale for next iteration.
    scaler.update()

    return loss.item()


def validate_segmentation(model, data, label, criterion, loss, classes):
    # Disable dropout and batch norm layers
    model.eval()

    # Don't compute gradients during evaluation step
    with torch.no_grad():
        # get model output and remove batch dimension
        predict = model(data).squeeze()
        # Get the prediction by getting the maximum probability along dimension 0
        predict = reverse_one_hot(predict)
        # predict = np.array(predict.detach().cpu())

        # get RGB label image and remove batch dimension
        label = label.squeeze()
        if loss == 'dice':
            label = reverse_one_hot(label)

        # label = np.array(label.detach().cpu())

        # compute per pixel accuracy
        precision = global_accuracy(predict, label)
        confusion_matrix = get_confusion_matrix(predict.flatten(), label.flatten(), classes)

        return precision, confusion_matrix
