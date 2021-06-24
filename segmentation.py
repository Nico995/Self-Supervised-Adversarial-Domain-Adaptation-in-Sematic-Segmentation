import torch
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dataset import camvid_data_loaders, idda_data_loaders
from model import BiSeNet
from methods.segmentation.training import training
from utils import load_segm_args, DiceLoss
from utils.loss import DiceLossV2, OhemCELoss


def main():
    """
    This is the TRAINING script entry point. Differently from most DL implementation, in the main() function we will only keep
    variables initializations and nothing else.
    We will make use of a script for the general training loop, called domain_adaptation.py (inside methods.py). Inside domain_adaptation.py one can find the
    basic loop structure (epochs and bacthes) common to all Deep Learning's model's tranining.
    The actual code concerning forward-pass, backpropagation and so on, will be in a separate script inside the
    "methods" package. Doing things in this way we hope to keep the code clear and readable, and avoid the (sadly too)
    common chains of if-elses that hide away the real code and make it appear way more complicated that what actually
    is.

    One should look here if they're looking for:
    - Dataset/Dataloader initialization
    - Model initialization
    - Optimizer, Learning Rate Scheduler & Criterion (loss fn.) initialization

    """

    # Read command line arguments
    args = load_segm_args()

    # build model
    model = BiSeNet(args.num_classes, args.context_path).cuda()

    # Get dataloader structures
    if args.dataset == 'CamVid':
        dataloader_train, dataloader_val = camvid_data_loaders(args.data, args.batch_size, args.num_workers, args.loss,
                                                               args.pre_encoded, args.crop_height, args.crop_width, do_augmentation=False)
    else:
        dataloader_train, dataloader_val = idda_data_loaders(args.data, args.batch_size, args.num_workers, args.loss,
                                                             args.pre_encoded, args.crop_height, args.crop_width, do_augmentation=False)

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)
    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # Loss function
    if args.loss == 'dice':
        criterion = DiceLoss()
        # criterion = DiceLossV2()
    elif args.loss == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'ohemce':
        criterion = OhemCELoss(0.7)
    else:
        NotImplementedError()
        exit()

    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

    # Add Gradscaler to prevent gradient underflowing under mixed precision training
    scaler = GradScaler()

    # Initialize scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=0.005)

    training(args, model, dataloader_train, dataloader_val, optimizer, scaler, criterion, scheduler)


if __name__ == '__main__':
    main()
