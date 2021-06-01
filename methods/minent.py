import numpy as np
import torch
from torch.cuda.amp import autocast

from utils import reverse_one_hot, global_accuracy, get_confusion_matrix


# TODO: loss must be crossentropy for this to work

def train_minent(model, source_images, source_labels, target_images, scaler, optimizer, source_criterion, adaptation_criterion):
    # Set model to Train mode
    model.train()

    # Clear optimizer gradient in an efficient way
    optimizer.zero_grad(set_to_none=True)

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

    ############
    #  Target  #
    # Training #
    ############
    with autocast():

        # Get network output
        output, output_sup1, output_sup2 = model(target_images)
        target_loss1 = adaptation_criterion(output)
        target_loss2 = adaptation_criterion(output_sup1)
        target_loss3 = adaptation_criterion(output_sup2)
        target_loss = target_loss1 + target_loss2 + target_loss3

    # Compute gradients with gradient scaler
    scaler.scale(target_loss).backward()

    scaler.step(optimizer)
    # Updates the scale for next iteration.
    scaler.update()

    return source_loss, target_loss


def validate_minent():
    pass
