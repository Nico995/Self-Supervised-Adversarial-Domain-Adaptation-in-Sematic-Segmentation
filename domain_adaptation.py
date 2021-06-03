import torch
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dataset import camvid_data_loaders, idda_data_loaders
from methods.adaptation.training import training
from model import BiSeNet
from utils import DiceLoss
from utils import load_da_args
from utils.loss import EntropyLoss


def main():
    """
    This is the TRAINING script entry point. Differently from most DL implementation, in the main() function we will only keep
    variables initializations and nothing else.
    We will make use of a script for the general training loop, called domain_adaptation.py. Inside domain_adaptation.py one can find the
    basic loop structure (epochs and bacthes) common to all Deep Learning's model's tranining.
    The actual code concerning forward-pass, backpropagation and so on, will be in a separate script inside the
    "methods" package. Doing things in this way we hope to keep the code clear and readable, and avoid the (sadly too)
    common chains of if-elses that hide away the real code and make it appear way more complicated that what actually
    is.

    One should look here if they're looking for:
    - Dataset/Dataloader initialization
    - Model initialization
    - Optimizer, Learning Rate Scheduler & Criterion(loss fn.) initialization

    """

    # Read command line arguments
    args = load_da_args()

    # build model
    model = BiSeNet(args.num_classes, args.context_path).cuda()

    dataloader_target_train, dataloader_target_val, len_images = \
        camvid_data_loaders(args.target_data, args.target_batch_size, args.num_workers, args.loss,
                            args.pre_encoded, args.crop_height, args.crop_width, shuffle=True, train_length=True,
                            do_augmentation=False)

    dataloader_source_train, dataloader_source_val = \
        idda_data_loaders(args.source_data, args.source_batch_size, args.num_workers, args.loss, args.pre_encoded,
                          args.crop_height, args.crop_width, shuffle=True, max_images=len_images, do_augmentation=False)

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)
    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # Loss function
    source_criterion = torch.nn.CrossEntropyLoss()
    adaptation_criterion = EntropyLoss()

    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

    # Add Gradscaler to prevent gradient underflowing under mixed precision training
    scaler = GradScaler()

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=0.005)

    # train
    training(args, model, optimizer, source_criterion, adaptation_criterion, scaler, scheduler,
             dataloader_source_train, dataloader_target_train, dataloader_source_val, dataloader_target_val)


if __name__ == '__main__':
    main()
