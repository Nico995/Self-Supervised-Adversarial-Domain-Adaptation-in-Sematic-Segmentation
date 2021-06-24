import numpy as np
import torch
from torch.cuda.amp import autocast

from utils import reverse_one_hot, global_accuracy, get_confusion_matrix
from utils.loss import bce_loss
from utils.utils import prob_2_entropy
from torch.nn.functional import softmax

# However, scaler.update should only be called once, after all optimizers used this iteration have been stepped:
# https://pytorch.org/docs/stable/notes/amp_examples.html

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def train_advent(model, main_discrim, aux_discrim, model_optimizer, main_discrim_optimizer, aux_discrim_optimizer,
                 source_images, source_labels, target_images, scaler, source_criterion, lambda_adv_main, lambda_adv_aux):

    # labels for adversarial training
    source_domain_label = 0
    target_domain_label = 1

    # Set model to Train mode
    model.train()

    # Clear optimizer gradient in an efficient way
    model_optimizer.zero_grad(set_to_none=True)
    main_discrim_optimizer.zero_grad(set_to_none=True)
    # aux_discrim_optimizer.zero_grad(set_to_none=True)

    ############
    #  Source  #
    # Training #
    ############

    # Move first set of images to GPU
    source_images, source_labels = source_images.cuda(), source_labels.cuda()

    # During source training, we don't want to accumulate gradients on the discriminators
    freeze(main_discrim)
    # freeze(aux_discrim)

    with autocast():
        # Get network output
        src_seg_out_main, src_seg_out_aux1, src_seg_out_aux2 = model(source_images)
        # Loss
        src_seg_loss_main = source_criterion(src_seg_out_main, source_labels)
        src_seg_loss_aux1 = source_criterion(src_seg_out_aux1, source_labels)
        src_seg_loss_aux2 = source_criterion(src_seg_out_aux2, source_labels)
        src_seg_loss = src_seg_loss_main + src_seg_loss_aux1 + src_seg_loss_aux2

    # Compute gradients with gradient scaler
    scaler.scale(src_seg_loss).backward()

    # Remove first set of images from GPU
    del source_images

    # Move second set of images to GPU
    target_images = target_images.cuda()

    # TODO: Finish implementing
    with autocast():
        # Get network output
        trg_seg_out_main, _, _ = model(target_images)
        trg_discrim_out = main_discrim(prob_2_entropy(softmax(trg_seg_out_main)))
        trg_adv_loss = bce_loss(trg_discrim_out, source_domain_label)

        loss = lambda_adv_main * trg_adv_loss

    # Compute gradients with gradient scaler
    scaler.scale(loss).backward()

    #Train discriminator networks
    unfreeze(main_discrim)
    # unfreeze(aux_discrim)
    
    # Train with source
    src_seg_out_main = src_seg_out_main.detach()

    with autocast():
        src_discrim_out = main_discrim(prob_2_entropy(softmax(src_seg_out_main)))
        src_discrim_loss = bce_loss(src_discrim_out, source_domain_label)
        src_discrim_loss = src_discrim_loss / 2

    scaler.scale(src_discrim_loss).backward()

    # Train with target
    trg_seg_out_main = trg_seg_out_main.detach()

    with autocast():
        trg_discrim_out = main_discrim(prob_2_entropy(softmax(trg_seg_out_main)))
        trg_discrim_loss = bce_loss(trg_discrim_out, target_domain_label)
        trg_discrim_loss = trg_discrim_loss / 2

    scaler.scale(trg_discrim_loss).backward()

    scaler.step(model_optimizer)
    scaler.step(main_discrim_optimizer)

    # Updates the scale for next iteration.
    scaler.update()

    return src_seg_loss.item(), trg_adv_loss.item(), src_discrim_loss.item(), trg_discrim_loss.item()


def validate_advent(model, data, label, criterion, loss, classes):
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