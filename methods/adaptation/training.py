import os

import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter

from methods import train_minent, validate_minent
from utils import intersection_over_union


def validation(args, model, dataloader_target_val, adaptation_criterion):
    """
    This function contains the validation loop
    """
    # Progress bar
    tq = tqdm.tqdm(total=len(dataloader_target_val) * args.target_batch_size)
    tq.set_description('val')

    # Metrics initialization
    running_entropy = []
    running_precision = []
    running_confusion_matrix = np.zeros((args.num_classes, args.num_classes))

    for i, (data, label) in enumerate(dataloader_target_val):
        # Move images to gpu
        data = data.cuda()
        label = label.cuda()

        '''
        This is the actual content of the VALIDATION loop.
        '''
        entropy, precision, confusion_matrix = validate_minent(model, data, label, adaptation_criterion, args.loss, args.num_classes)

        # Store metrics
        running_entropy.append(entropy)
        running_precision.append(precision)
        running_confusion_matrix += confusion_matrix

        # Progress bar
        tq.update(args.target_batch_size)

    entropy = np.mean(running_entropy)
    precision = np.mean(running_precision)
    per_class_iou, mean_iou = intersection_over_union(confusion_matrix)

    print('Entropy [eval]: %.3f' % entropy)
    print('Global Precision [eval]: %.3f' % precision)
    print('Mean IoU [eval]: %.3f' % mean_iou)
    print('Per Class IoU [eval]: \n', list(per_class_iou))
    tq.close()

    return entropy, precision, mean_iou

def training(args, model, optimizer, source_criterion, adaptation_criterion, scaler, scheduler, dataloader_source_train, dataloader_target_train,
             dataloader_source_val, dataloader_target_val):
    """
    This is the common double-loop structure that most of us are familiar with.
    One must look here if they're looking for:
    - Metric aggregation
    - Logging and Tensorboard variables
    - Progress bar
    - Images/Labels going to GPU (.cuda())
    - Checkpointing
    - Validation Entry-point
    """
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    # initialize temp variables
    best_mean_iou = 0
    step = 0

    # Epoch loop
    for epoch in range(args.num_epochs):
        # Progress bar
        tq = tqdm.tqdm(total=len(dataloader_source_train) * args.source_batch_size)
        tq.set_description('epoch %d, lr %.3f' % (epoch + 1, scheduler.state_dict()["_last_lr"][0]))

        running_source_loss = []
        running_target_loss = []
        # Batch loop
        for i, (source_data, target_data) in enumerate(zip(dataloader_source_train, dataloader_target_train)):
            # Move images to gpu
            source_data, source_label = source_data[0], source_data[1]
            target_data = target_data[0]

            '''
            This is the actual content of the TRAINING loop.
            '''
            source_loss, target_loss = train_minent(model, source_data, source_label, target_data, optimizer, scaler, source_criterion, adaptation_criterion)

            # Logging & progress bar
            step += 1
            writer.add_scalar('source_loss_step', source_loss, step)
            writer.add_scalar('target_loss_step', target_loss, step)
            running_source_loss.append(source_loss)
            running_target_loss.append(target_loss)

            tq.update(args.source_batch_size)
            tq.set_postfix({'source_loss': f'{source_loss:.6f}', 'target_loss': f'{target_loss:.6f}'})

        # Update learning rate at the end of each batch
        scheduler.step()

        # Logging & progress bar
        source_loss_train_mean = np.mean(running_source_loss)
        target_loss_train_mean = np.mean(running_target_loss)
        writer.add_scalar('epoch/source_loss_train_mean', float(source_loss_train_mean), epoch)
        writer.add_scalar('epoch/target_loss_train_mean', float(target_loss_train_mean), epoch)

        tq.set_postfix({'source_loss_train_mean': f'{source_loss_train_mean:.6f}',
                        'target_loss_train_mean': f'{target_loss_train_mean:.6f}'})
        tq.close()

        # Validation step
        if (epoch + 1) % args.validation_step == 0 or epoch == args.num_epochs:
            '''
            This is the actual content of the validation loop
            '''
            entropy, precision, mean_iou = validation(args, model, dataloader_target_val, adaptation_criterion)
            # Save model if has better accuracy
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                # Save weights
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.save_model_path, 'best_dice_loss.pth'))

            # Tensorboard Logging
            writer.add_scalar('epoch/entropy', entropy, epoch)
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', mean_iou, epoch)

        # Recurrent Checkpointing
        if (epoch + 1) % args.checkpoint_step == 0 or epoch == args.num_epochs:
            # Save weights
            os.makedirs(args.save_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_model_path, 'latest_dice_loss.pth'))
