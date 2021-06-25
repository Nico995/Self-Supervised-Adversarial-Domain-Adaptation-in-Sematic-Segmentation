import argparse


def load_segm_args():
    # basic parameters
    parser = argparse.ArgumentParser()

    # Training config
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='learning rate used for train')

    # Loss config
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')

    # Checkpoint config
    parser.add_argument('--checkpoint_step', type=int, default=20, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')

    # Crop config
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')

    # Model architecture config
    parser.add_argument('--context_path', type=str, default="resnet18", help='The context path backbone model to use.')
    parser.add_argument('--num_classes', type=int, default=12, help='num of object classes (with void)')

    # Save path config
    parser.add_argument('--data', type=str, default='data/CamVid/', help='path of training data')
    parser.add_argument('--save_model_path', type=str, default='./checkpoints', help='path to save model')

    # Environment resources config
    parser.add_argument('--num_workers', type=int, default=6, help='num of workers')
    parser.add_argument('--use_gpu', action='store_true', help='whether to user gpu for training')

    # Misc
    parser.add_argument('--pre_encoded', action='store_true', help='whether to use pre encoded labels or not')
    parser.add_argument('--dataset', type=str, default="CamVid", help='name of the dataset ued (CamVid/IDDA?)')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')

    args = parser.parse_args()

    return args


def load_da_args():

    # basic parameters
    parser = argparse.ArgumentParser()

    # Training config
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='learning rate used for train')
    parser.add_argument('--learning_rate_disc', type=float, default=0.0001, help='learning rate used for train')
    parser.add_argument('--source_batch_size', type=int, default=2, help='Number of images in each batch')
    parser.add_argument('--target_batch_size', type=int, default=2, help='Number of images in each batch')

    parser.add_argument('--source_data', type=str, default='data/IDDA/', help='path of source training data')
    parser.add_argument('--target_data', type=str, default='data/CamVid/', help='path of target training data')

    # Crop config
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')

    # Loss config
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')

    # Checkpoint config
    parser.add_argument('--checkpoint_step', type=int, default=20, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')

    # Environment resources config
    parser.add_argument('--num_workers', type=int, default=6, help='num of workers')
    parser.add_argument('--use_gpu', action='store_true', help='whether to user gpu for training')

    # Model architecture config
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path backbone model to use.')
    parser.add_argument('--num_classes', type=int, default=12, help='num of object classes (with void)')

    # Misc
    parser.add_argument('--pre_encoded', action='store_true', help='whether to use pre encoded labels or not')
    parser.add_argument('--save_model_path', type=str, default='./checkpoints', help='path to save model')

    args = parser.parse_args()

    return args
