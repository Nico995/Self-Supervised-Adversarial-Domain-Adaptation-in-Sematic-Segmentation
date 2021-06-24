import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F, CenterCrop
import albumentations as a
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, Resize

from utils import encode_label_crossentropy, encode_label_dice, get_label_info
from utils.custom_transforms import RandomGaussianBlur, ColorDistortion, RandomDiscreteScale


class ToChannelLast(object):
    """
    Custom transformation to convert standard tensor dimensions (C, W, H) to convenient dimensions (W, H, C)
    """

    def __init__(self):
        pass

    def __call__(self, image):
        return image.permute(1, 2, 0)


class CamVid(torch.utils.data.Dataset):
    """
    Custom dataset class to manage images and labels
    """

    def __init__(self, image_path, label_path, csv_path, image_size, mode='train', loss='dice', pre_encoded=False, do_augmentation=True):
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
        self.scales = [1, 1.25, 1.5, 1.75, 2]
        self.pre_encoded = pre_encoded
        self.do_augmentation = do_augmentation

        self.mode = mode

        # Read metadata
        self.label_info = get_label_info(csv_path)

        # Get images folders paths (as list)
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, f'*.png')))
        self.image_list.sort()

        # Get labels folders paths (as list)
        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            if not self.pre_encoded:
                self.label_list.extend(glob.glob(os.path.join(label_path_, f'*.png')))
            else:
                self.label_list.extend(glob.glob(os.path.join(label_path_, f'*.npy')))
        self.label_list.sort()

        self.transform = a.Compose([
            a.HorizontalFlip(p=0.5),
            RandomDiscreteScale(self.scales, p=1),
            a.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_WRAP),
            a.RandomCrop(image_size[0], image_size[1], p=1)
        ])

        self.augment = Compose([ColorDistortion(), RandomGaussianBlur()])

        # Transformations
        self.normalize = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        # Read images and transform
        image = Image.open(self.image_list[index])

        # Labels are standard rgb images
        label = Image.open(self.label_list[index])

        if self.mode == 'train':
            trn = self.transform(image=np.array(image), mask=np.array(label))
            image, label = trn['image'], trn['mask']

        if self.mode == 'train' and self.do_augmentation:
            image = self.augment(image)

        # Encode label
        if self.loss == 'dice':
            # Encode label image
            label = encode_label_dice(label, self.label_info).astype(np.uint8)
            label = torch.from_numpy(label)

        elif self.loss == 'crossentropy':
            # Encode label image
            label = encode_label_crossentropy(label, self.label_info).astype(np.uint8)
            label = torch.from_numpy(label).long()

        image = self.normalize(image).float()
        return image, label

    def __len__(self):
        return len(self.image_list)


def get_data_loaders(data, batch_size, num_workers, loss, pre_encoded, crop_height, crop_width, shuffle=True, train_length=False, do_augmentation=True):
    """
    Build dataloader structures for train and validation

    Args:
        args: command line arguments

    Returns:
        dataloader structures for train and validation
    """

    # Build paths
    train_path = [os.path.join(data, 'train'), os.path.join(data, 'val')]
    train_label_path = [os.path.join(data, 'train_labels'), os.path.join(data, 'val_labels')]

    val_path = os.path.join(data, 'test')
    val_label_path = os.path.join(data, 'test_labels')

    csv_path = os.path.join(data, 'class_dict.csv')
    # Train Dataloader
    dataset_train = CamVid(train_path, train_label_path, csv_path, (crop_height, crop_width),
                           loss=loss, pre_encoded=pre_encoded, mode='train', do_augmentation=do_augmentation)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle,
    )

    # Val Dataloader
    dataset_val = CamVid(val_path, val_label_path, csv_path, (crop_height, crop_width),
                         loss=loss, mode='val', pre_encoded=pre_encoded, do_augmentation=False)
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        num_workers=num_workers,
        shuffle=shuffle
    )

    if train_length:
        return dataloader_train, dataloader_val, len(dataset_train)
    else:
        return dataloader_train, dataloader_val

