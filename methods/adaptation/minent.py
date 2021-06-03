import numpy as np
import torch
from torch.cuda.amp import autocast

from utils import reverse_one_hot, global_accuracy, get_confusion_matrix


def train_minent(model, source_images, source_labels, target_images, optimizer, scaler, source_criterion, adaptation_criterion):
    source_labels = torch.argmax(source_labels, dim=1).long()

    # Set model to Train mode
    model.train()

    # Clear optimizer gradient in an efficient way
    optimizer.zero_grad(set_to_none=True)

    # Move first set of images to GPU
    source_images, source_labels = source_images.cuda(), source_labels.cuda()
    ############
    #  Source  #
    # Training #
    ############
    with autocast():

        # Get network output
        output, output_sup1, output_sup2 = model(source_images)
        # Loss
        source_loss1 = source_criterion(output, source_labels)
        source_loss2 = source_criterion(output_sup1, source_labels)
        source_loss3 = source_criterion(output_sup2, source_labels)
        source_loss = source_loss1 + source_loss2 + source_loss3

    # Compute gradients with gradient scaler
    scaler.scale(source_loss).backward()

    # Remove first set of images from GPU
    del source_images
    del source_labels
    # Move second set of images to GPU
    target_images = target_images.cuda()

    ############
    #  Target  #
    # Training #
    ############
    with autocast():

        # Get network output
        output, _, _ = model(target_images)
        target_loss = adaptation_criterion(output)
        # target_loss2 = adaptation_criterion(output_sup1)
        # target_loss3 = adaptation_criterion(output_sup2)
        # target_loss = target_loss1 + target_loss2 + target_loss3

    # Compute gradients with gradient scaler
    scaler.scale(target_loss).backward()

    scaler.step(optimizer)
    # Updates the scale for next iteration.
    scaler.update()

    optimizer.zero_grad()

    return source_loss.item(), target_loss.mean().item()


def validate_minent(model, data, label, criterion, loss, classes):
    # Disable dropout and batch norm layers
    model.eval()

    # Don't compute gradients during evaluation step
    with torch.no_grad():
        ############
        #  Target  #
        #    Val   #
        ############
        with autocast():

            # Get network output
            output = model(data)
            target_loss = criterion(output)

        # get model output
        predict = output.squeeze()
        predict = reverse_one_hot(predict)
        predict = np.array(predict.detach().cpu())

        # get RGB label image
        label = label.squeeze()
        label = reverse_one_hot(label)
        label = np.array(label.detach().cpu())

        # compute per pixel accuracy
        precision = global_accuracy(predict, label)
        confusion_matrix = get_confusion_matrix(label.flatten(), predict.flatten(), classes)

        return target_loss.item(), precision, confusion_matrix