import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, Resize

from utils import encode_label_crossentropy, encode_label_dice, get_label_info


class ToChannelLast(object):
    """
    Custom transformation to convert standard tensor dimensions (C, W, H) to convenient dimensions (W, H, C)
    """

    def __init__(self):
        pass

    def __call__(self, image):
        return image.permute(1, 2, 0)


# TODO: Remove and substitute with a Transform into the Compose (GaussianBlur is already implemented in pytorch)
def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass


def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass


class CamVid(torch.utils.data.Dataset):
    """
    Custom dataset class to manage images and labels
    """

    def __init__(self, image_path, label_path, csv_path, image_size, loss='dice'):
        """

        Args:
            image_path (str): path of the images folder.
            label_path (str): path of the labels folder.
            csv_path (str): path for the csv metadata.
            image_size (tuple): image width and height.
            loss (str): loss to utilize (changes the way that labels are transformed)
        """

        super().__init__()
        self.image_size = image_size
        self.loss = loss
        self.scales = [0.5, 1, 1.25, 1.5, 1.75, 2]

        # Read metadata
        self.label_info = get_label_info(csv_path)

        # Get images folders paths (as list)
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.png')))
        self.image_list.sort()

        # Get labels folders paths (as list)
        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()

        # Transformations
        self.normalize = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        # Seed transformations (have to be the same random crop for image and label)
        seed = int(random.random())
        torch.random.manual_seed(seed)
        random.seed(seed)

        # Choose a random scale
        random_scale = random.choice(self.scales)
        scaled_image_size = (int(self.image_size[0] * random_scale), int(self.image_size[1] * random_scale))

        # Read images and transform
        image = Image.open(self.image_list[index])
        image = Resize(scaled_image_size)(image)
        image = RandomCrop(self.image_size, seed, pad_if_needed=True)(image)
        image = self.normalize(image).float()

        # Read labels and transform
        label = Image.open(self.label_list[index])
        label = Resize(scaled_image_size)(label)
        label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)

        # Encode label
        if self.loss == 'dice':
            # Encode label image
            label = encode_label_dice(label, self.label_info).astype(np.uint8)
            label = torch.from_numpy(label)

        elif self.loss == 'crossentropy':
            # Encode label image
            label = encode_label_crossentropy(label, self.label_info).astype(np.uint8)
            label = torch.from_numpy(label).long()

        return image, label

    def __len__(self):
        return len(self.image_list)


def get_data_loaders(args):
    """
    Build dataloader structures for train and validation

    Args:
        args: command line arguments

    Returns:
        dataloader structures for train and validation
    """

    # Build paths
    train_path = [os.path.join(args.data, 'train'), os.path.join(args.data, 'val')]
    train_label_path = [os.path.join(args.data, 'train_labels'), os.path.join(args.data, 'val_labels')]

    test_path = os.path.join(args.data, 'test')
    test_label_path = os.path.join(args.data, 'test_labels')

    csv_path = os.path.join(args.data, 'class_dict.csv')

    # Train Dataloader
    dataset_train = CamVid(train_path, train_label_path, csv_path, image_size=(args.crop_height, args.crop_width),
                           loss=args.loss)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # Val Dataloader
    dataset_val = CamVid(test_path, test_label_path, csv_path, image_size=(args.crop_height, args.crop_width),
                         loss=args.loss)
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        num_workers=args.num_workers,
    )

    return dataloader_train, dataloader_val
