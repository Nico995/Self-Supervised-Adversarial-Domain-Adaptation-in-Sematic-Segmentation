import os

import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast

from dataset import get_data_loaders
from model import BiSeNet
from utils import load_args, poly_lr_scheduler, DiceLoss
from eval import validate


def train(args, model, optimizer, criterion, scaler, dataloader_train, dataloader_val, csv_path):
    """
    Training loop for the model.

    Args:
        args: command line arguments.
        model: PyTorch model to train.
        optimizer: Optimizer to use during training.
        criterion: Loss function.
        scaler: Gradient scaler, necessary to train with AMP.
        dataloader_train: Dataloader for training data.
        dataloader_val: Dataloader for validation data.
        csv_path: path to csv metadata

    Returns:
        Nothing
    """
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    # initialize temp variables
    best_mean_iou = 0
    step = 0

    # Epoch loop
    for epoch in range(args.num_epochs):
        # Get learning rate
        lr = poly_lr_scheduler(optimizer, starting_lr=args.learning_rate, current_iter=epoch, max_iter=args.num_epochs)

        # Set model to Train mode
        model.train()

        # Progress bar
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))

        loss_record = []
        # Batch loop
        for i, (data, label) in enumerate(dataloader_train):
            # Move images to gpu
            if args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            with autocast():
                # Get network output
                output, output_sup1, output_sup2 = model(data)

                # Loss
                loss1 = criterion(output, label)
                loss2 = criterion(output_sup1, label)
                loss3 = criterion(output_sup2, label)
                loss = loss1 + loss2 + loss3

            # Clear optimizer gradient in an efficient way
            optimizer.zero_grad(set_to_none=True)

            # Compute gradients with gradient scaler
            # scaler.scale(loss).backward()

            # Compute gradients
            loss.backward()

            # scaler.step(optimizer)
            # Updates the scale for next iteration.
            # scaler.update()

            # Back-propagate gradients
            optimizer.step()

            # Logging & progress bar
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

        # Logging & progress bar
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)

        tq.set_postfix(mean_loss='%.6f' % loss_train_mean)
        tq.close()

        # Checkpointing
        if (epoch + 1) % args.checkpoint_step == 0 or epoch == args.num_epochs:
            # Build output dir
            os.makedirs(args.save_model_path, exist_ok=True)
            # Save weights
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest_dice_loss.pth'))

        # Validation step
        if (epoch + 1) % args.validation_step == 0 or epoch == args.num_epochs:
            precision, mean_iou = validate(args, model, dataloader_val, csv_path)
            # Save model if has better accuracy
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                # Build output dir
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best_dice_loss.pth'))

            # Tensorboard Logging
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', mean_iou, epoch)


def main():
    # Read command line arguments
    args = load_args()
    csv_path = os.path.join(args.data, 'class_dict.csv')

    # Get dataloader structures
    dataloader_train, dataloader_val = get_data_loaders(args)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # load pretrained model if exists
    if args.pretrained_model_path:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # Loss function
    if args.loss == 'dice':
        criterion = DiceLoss()
    else:  # crossentropy
        criterion = torch.nn.CrossEntropyLoss()

    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

    # Add Gradscaler to prevent gradient underflowing under mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # train
    train(args, model, optimizer, criterion, scaler, dataloader_train, dataloader_val, csv_path)


if __name__ == '__main__':
    main()
