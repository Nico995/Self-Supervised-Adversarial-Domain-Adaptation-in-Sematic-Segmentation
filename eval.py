from dataset.CamVid import CamVid
import torch
import argparse
import os
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils import reverse_one_hot, global_accuracy, get_confusion_matrix, intersection_over_union
import tqdm


def validate(args, model, dataloader, csv_path):
    """
    Perform a validation epoch.

    Args:
        args (Namespace): Command line arguments.
        model: Model to evaluate.
        dataloader (DataLoader): Validation data loader.
        csv_path (str): Path to csv metadata

    Returns:
        Metrics
    """

    # Don't compute gradients during evaluation step
    with torch.no_grad():
        # Disable dropout and batch norm layers
        model.eval()

        # Progress bar
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')

        precision_record = []

        confusion_matrix = np.zeros((args.num_classes, args.num_classes))
        # Batch loop
        for i, (data, label) in enumerate(dataloader):
            if i == 240:
                break

            # Move images to gpu
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get model output
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.detach().cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.detach().cpu())

            # compute per pixel accuracy
            precision = global_accuracy(predict, label)
            confusion_matrix += get_confusion_matrix(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

            # Progress bar
            tq.update(args.batch_size)

        precision = np.mean(precision_record)
        per_class_iou, mean_iou = intersection_over_union(confusion_matrix, csv_path)

        print('Global Precision [eval]: %.3f' % precision)
        print('Mean IoU [eval]: %.3f' % mean_iou)
        print('Per Class IoU [eval]: \n', list(per_class_iou))
        tq.close()

        return precision, mean_iou
