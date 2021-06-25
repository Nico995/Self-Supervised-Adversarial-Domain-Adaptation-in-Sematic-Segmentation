import os

import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter

from methods import validate_segmentation, train_segmentation
from utils import intersection_over_union, plot_prediction
from utils.utils import poly_lr_scheduler


classes = ['Byc', 'Bld', 'Car', 'Pol', "Fnc", "Ped", "Rod", "Sdw", "Sin", "Sky", "Tre"]


def validation(args, model, dataloader_val, criterion):
    """
    This function contains the validation loop
    """

    # Progress bar
    tq = tqdm.tqdm(total=len(dataloader_val))
    tq.set_description('val')

    # Metrics initialization
    running_precision = []
    running_confusion_matrix = []

    for i, (data, label) in enumerate(dataloader_val):
        # Move images to gpu
        data = data.cuda()
        label = label.cuda()

        '''
        This is the actual content of the VALIDATION loop.
        '''
        precision, confusion_matrix = validate_segmentation(model, data, label, criterion, args.loss, args.num_classes)

        # Store metrics
        running_precision.append(precision)
        running_confusion_matrix.append(confusion_matrix)

        # Progress bar
        tq.update(1)

        # Early stopping when training IDDA
        if i >= 233:
            break

    precision = np.mean(running_precision)
    per_class_iou, mean_iou = intersection_over_union(torch.stack(running_confusion_matrix).sum(dim=0))

    tq.close()

    return precision, per_class_iou, mean_iou


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

        ###
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        ###

        # Progress bar
        tq = tqdm.tqdm(total=len(dataloader_train))
        # tq.set_description('epoch %d, lr %.3f' % (epoch + 1, scheduler.state_dict()["_last_lr"][0]))
        tq.set_description('epoch %d, lr %.3f' % (epoch + 1, lr))

        loss_record = []

        # Batch loop
        for i, (data, label) in enumerate(dataloader_train):
            # Uncomment this to visualize the batch images and labels
            # plt.imshow(batch_to_plottable_image(data))
            # plt.show()
            # plt.imshow(label_to_plottable_image(label))
            # plt.show()
            # # Get class readable name from RGB color
            # # Plot section (again, this is wrong, the section should be circular)
            # plt.imshow(batch_to_plottable_image(data))
            # plt.imshow(label_to_plottable_image(label), alpha=0.5)
            # plt.show()
            # exit()

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

            tq.update(1)
            tq.set_postfix(loss='%.6f' % loss)

            # Early stopping when training IDDA
            if i >= 78:
                break

            # Update learning rate at the end of each batch
        # scheduler.step()

        # Logging & progress bar
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)

        tq.set_postfix(mean_loss='%.6f' % loss_train_mean)
        tq.close()

        # Plot prediction image
        plot_prediction(model, dataloader_val, epoch, args.dataset)

        # Validation step
        if (epoch + 1) % args.validation_step == 0 or epoch == args.num_epochs:
            '''
            This is the actual content of the validation loop
            '''
            precision, per_class_iou, mean_iou = validation(args, model, dataloader_val, criterion)
            # Save model if has better accuracy
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                # Save weights
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.save_model_path, 'best_loss.pth'))

            for cls, iou in zip(classes, per_class_iou):
                writer.add_scalar(f'epoch/{cls}_iou', iou, epoch)

            # Tensorboard Logging
            writer.add_scalar('epoch/precision', precision, epoch)
            writer.add_scalar('epoch/miou', mean_iou, epoch)

            print(f'global precision:\t\t {precision:.3f}')
            print(f'mean iou:\t\t {mean_iou:.3f}')
            print(f"per-class iou:\n {[f'{cls}: {iou:.3f}' for cls, iou in zip(classes, per_class_iou)]}")

        # Recurrent Checkpointing
        if (epoch + 1) % args.checkpoint_step == 0 or epoch == args.num_epochs:
            # Save weights
            os.makedirs(args.save_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_model_path, 'latest_loss.pth'))
