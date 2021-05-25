import glob
import os
import random

import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, Resize

from utils import encode_label_crossentropy, get_label_info, encode_label_idda_dice
from utils.custom_transforms import RandomGaussianBlur, ColorDistortion


class IDDA(torch.utils.data.Dataset):
    """
    Custom dataset class to manage images and labels
    """

    def __init__(self, image_path, label_path, csv_path, image_size, mode='train', loss='dice', pre_encoded=False):
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
        self.pre_encoded = pre_encoded

        self.mode = mode  # Read metadata

        self.label_info = get_label_info(csv_path)

        # Get images folders paths (as list)
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, f'*.jpg')))
        self.image_list.sort()

        # Get labels folders paths (as list)
        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            if not self.pre_encoded:
                self.label_list.extend(glob.glob(os.path.join(label_path_, f'*.png')))
            else:
                self.label_list.extend(glob.glob(os.path.join(label_path_, f'*.npz')))

        self.label_list.sort()

        # Transformations
        self.normalize = Compose([
            ToTensor(),
            Normalize((0.39068785, 0.40521392, 0.41434407), (0.29652068, 0.30514979, 0.30080369)),
        ])

        self.augment = Compose([ColorDistortion(), RandomGaussianBlur()])

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

        # Read encode and transform label
        if not self.pre_encoded:
            # Labels are standard rgb images
            label = Image.open(self.label_list[index]).convert("RGB")
            label = Resize(scaled_image_size)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)

            # Encode label
            if self.loss == 'dice':
                # Encode label image
                label = encode_label_idda_dice(label).astype(np.uint8)
                label = torch.from_numpy(label)

            elif self.loss == 'crossentropy':
                # TODO: Implement crossentropy encoding
                print("not implemented yet")
                exit('-1')
        else:
            # Labels are numpy.ndarrays
            label = np.moveaxis(np.load(self.label_list[index])["a"], -1, 0)
            label = torch.tensor(label)
            label = Resize(scaled_image_size)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)

        flip = torch.rand(1)

        # if flip < 0.75:
        #     image = F.hflip(image)
        #     label = F.hflip(label)

        # image = self.augment(image)

        image = self.normalize(image).float()
        return image, label

    def __len__(self):
        return len(self.image_list)


def get_data_loaders(args, shuffle=False):
    """
    Build dataloader structures for train and validation

    Args:
        args: command line arguments

    Returns:
        dataloader structures for train and validation
    """

    # Build paths
    train_path = [os.path.join(args.data, 'train')]
    train_label_path = [os.path.join(args.data, 'train_labels')]

    test_path = os.path.join(args.data, 'val')
    test_label_path = os.path.join(args.data, 'val_labels')

    csv_path = os.path.join(args.data, 'class_dict.csv')
    # Train Dataloader
    dataset_train = IDDA(train_path, train_label_path, csv_path, (args.crop_height, args.crop_width),
                           loss=args.loss, pre_encoded=args.pre_encoded)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=shuffle
    )

    # Val Dataloader
    dataset_val = IDDA(test_path, test_label_path, csv_path, (args.crop_height, args.crop_width),
                         loss=args.loss, mode='val', pre_encoded=args.pre_encoded)
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=shuffle
    )

    return dataloader_train, dataloader_val
