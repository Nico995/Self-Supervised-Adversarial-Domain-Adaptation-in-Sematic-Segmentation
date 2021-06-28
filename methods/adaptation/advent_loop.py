import os

import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter

from utils import intersection_over_union
from utils.utils import poly_lr_scheduler
from .advent_training import advent_training
from .advent_validation import advent_validation

classes = ['Byc', 'Bld', 'Car', 'Pol', "Fnc", "Ped", "Rod", "Sdw", "Sin", "Sky", "Tre"]


def validation(args, model, dataloader_target_val, segmentation_criterion):
    """
    This function contains the validation loop
    """
    # Progress bar
    tq = tqdm.tqdm(total=len(dataloader_target_val))
    tq.set_description('val')

    # Metrics initialization
    running_precision = []
    running_confusion_matrix = []

    for i, (data, label) in enumerate(dataloader_target_val):
        # Move images to gpu
        data = data.cuda()
        label = label.cuda()

        '''
        This is the actual content of the VALIDATION loop.
        '''
        precision, confusion_matrix = advent_validation(model, data, label, segmentation_criterion, args.loss,
                                                        args.num_classes)

        # Store metrics
        running_precision.append(precision)
        running_confusion_matrix.append(confusion_matrix)

        # Progress bar
        tq.update(1)

    precision = np.mean(running_precision)

    per_class_iou, mean_iou = intersection_over_union(torch.stack(running_confusion_matrix).sum(dim=0))

    tq.close()

    return precision, per_class_iou, mean_iou


def training(args, model, main_discrim, aux_discrim, model_optimizer, main_discrim_optimizer, aux_discrim_optimizer,
             segmentation_criterion, adversarial_criterion, scaler, dataloader_source_train, dataloader_target_train,
             dataloader_target_val, lambda_adv_main, lambda_adv_aux):
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
        lr_train = poly_lr_scheduler(model_optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        lr_discrim_main = poly_lr_scheduler(main_discrim_optimizer, args.learning_rate_disc, iter=epoch,
                                            max_iter=args.num_epochs)
        # lr_discrim_aux = poly_lr_scheduler(aux_discrim_optimizer, args.learning_rate_disc, iter=epoch,
        # max_iter=args.num_epochs)
        ###

        # Progress bar
        tq = tqdm.tqdm(total=len(dataloader_source_train))
        # tq.set_description(f'epoch {epoch + 1}, lr: [m:{lr_train:.3f} dm:{lr_discrim_main:.3f} da:{lr_discrim_aux:.3f}')
        tq.set_description(f'epoch {epoch + 1}, lr: [m:{lr_train:.3f} dm:{lr_discrim_main:.3f}')

        running_src_seg_loss = []
        running_trg_adv_loss = []
        running_src_discrim_loss = []
        running_trg_discrim_loss = []
        # Batch loop
        for i, (source_data, target_data) in enumerate(zip(dataloader_source_train, dataloader_target_train)):
            # Move images to gpu
            source_images, source_labels = source_data[0], source_data[1]
            target_images = target_data[0]
            '''
            This is the actual content of the TRAINING loop.
            '''
            src_seg_loss, trg_adv_loss, src_discrim_loss, trg_discrim_loss = \
                advent_training(model, main_discrim, aux_discrim, model_optimizer, main_discrim_optimizer,
                                aux_discrim_optimizer, source_images, source_labels, target_images,
                                scaler, segmentation_criterion, adversarial_criterion, lambda_adv_main, lambda_adv_aux)

            # Logging & progress bar
            step += 1
            writer.add_scalar('src_seg_loss', src_seg_loss, step)
            writer.add_scalar('trg_adv_loss', trg_adv_loss, step)
            writer.add_scalar('src_discrim_loss', src_discrim_loss, step)
            writer.add_scalar('trg_discrim_loss', trg_discrim_loss, step)
            running_src_seg_loss.append(src_seg_loss)
            running_trg_adv_loss.append(trg_adv_loss)
            running_src_discrim_loss.append(src_discrim_loss)
            running_trg_discrim_loss.append(trg_discrim_loss)

            tq.update(1)
            tq.set_postfix({'src_seg_loss': f'{src_seg_loss:.6f}', 'trg_adv_loss': f'{trg_adv_loss:.6f}',
                            'src_discrim_loss': f'{src_discrim_loss:.6f}',
                            'trg_discrim_loss': f'{trg_discrim_loss:.6f}'})

        # Logging & progress bar
        src_seg_loss_mean = np.mean(running_src_seg_loss)
        trg_adv_loss_mean = np.mean(running_trg_adv_loss)
        src_discrim_loss_mean = np.mean(running_src_discrim_loss)
        trg_discrim_loss_mean = np.mean(running_trg_discrim_loss)

        writer.add_scalar('epoch/src_seg_loss_mean', float(src_seg_loss_mean), epoch)
        writer.add_scalar('epoch/trg_adv_loss_mean', float(trg_adv_loss_mean), epoch)
        writer.add_scalar('epoch/src_discrim_loss_mean', float(src_discrim_loss_mean), epoch)
        writer.add_scalar('epoch/trg_discrim_loss_mean', float(trg_discrim_loss_mean), epoch)

        tq.set_postfix({'src_seg_loss_mean': f'{src_seg_loss_mean:.6f}',
                        'trg_adv_loss_mean': f'{trg_adv_loss_mean:.6f}',
                        'src_discrim_loss_mean': f'{src_discrim_loss_mean:.6f}',
                        'trg_discrim_loss_mean': f'{trg_discrim_loss_mean:.6f}'})
        tq.close()

        # Validation step
        if (epoch + 1) % args.validation_step == 0 or epoch == args.num_epochs:
            '''
            This is the actual content of the validation loop
            '''
            precision, per_class_iou, mean_iou = validation(args, model, dataloader_target_val, segmentation_criterion)
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
