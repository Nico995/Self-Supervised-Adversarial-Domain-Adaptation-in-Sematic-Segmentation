import os

import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter

from methods import validate_segmentation, train_segmentation
from utils import intersection_over_union


def validation(args, model, dataloader_val, criterion):
    """
    This function contains the validation loop
    """

    # Progress bar
    tq = tqdm.tqdm(total=len(dataloader_val) * args.batch_size)
    tq.set_description('test')

    # Metrics initialization
    running_precision = []
    running_confusion_matrix = np.zeros((args.num_classes, args.num_classes))

    for i, (data, label) in enumerate(dataloader_val):
        # Move images to gpu
        data = data.cuda()
        label = label.cuda()

        '''
        This is the actual content of the VALIDATION loop.
        '''
        precision, confusion_matrix = validate_segmentation(model, data, label, criterion, args.loss, args.num_classes)

        # Store metrics
        running_confusion_matrix += confusion_matrix
        running_precision.append(precision)

        # Progress bar
        tq.update(args.batch_size)

    precision = np.mean(running_precision)
    per_class_iou, mean_iou = intersection_over_union(confusion_matrix)

    print('Global Precision [eval]: %.3f' % precision)
    print('Mean IoU [eval]: %.3f' % mean_iou)
    print('Per Class IoU [eval]: \n', list(per_class_iou))
    tq.close()

    return precision, mean_iou


def training(args, model, dataloader_train, dataloader_val, optimizer, scaler, criterion, scheduler):
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
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %.3f' % (epoch + 1, scheduler.state_dict()["_last_lr"][0]))

        loss_record = []
        # Batch loop
        for i, (data, label) in enumerate(dataloader_train):
            # Move images to gpu
            data = data.cuda()
            label = label.cuda()

            '''
            This is the actual content of the TRAINING loop.
            '''
            loss = train_segmentation(model, data, label, optimizer, scaler, criterion, args.loss)

            # Logging & progress bar
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss)

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

        # Update learning rate at the end of each batch
        scheduler.step()

        # Logging & progress bar
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)

        tq.set_postfix(mean_loss='%.6f' % loss_train_mean)
        tq.close()

        # Validation step
        if (epoch + 1) % args.validation_step == 0 or epoch == args.num_epochs:
            '''
            This is the actual content of the validation loop
            '''
            precision, mean_iou = validation(args, model, dataloader_val, criterion)
            # Save model if has better accuracy
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                # Save weights
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.save_model_path, 'best_dice_loss.pth'))

            # Tensorboard Logging
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', mean_iou, epoch)

        # Recurrent Checkpointing
        if (epoch + 1) % args.checkpoint_step == 0 or epoch == args.num_epochs:
            # Save weights
            os.makedirs(args.save_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_model_path, 'latest_dice_loss.pth'))
