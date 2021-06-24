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

from utils import encode_label_crossentropy, encode_label_dice, get_label_info, encode_label_idda_dice
from utils.custom_transforms import RandomGaussianBlur, ColorDistortion, RandomDiscreteScale
from utils.utils import encode_label_idda_crossentropy


class IDDA(torch.utils.data.Dataset):
    """
    Custom dataset class to manage images and labels
    """

    def __init__(self, image_path, label_path, csv_path, image_size, mode='train', loss='dice', pre_encoded=False,
                 max_images=None, do_augmentation=True):
        """

        Args:
            image_path (str): path of the images folder.
            label_path (str): path of the labels folder.
            csv_path (str): path for the csv metadata.
            image_size (tuple): image width and height.
            loss (str): loss to utilize (changes the way that labels are transformed)
        """

        super().__init__()
        # Store some parameters
        self.image_size = image_size
        self.loss = loss
        self.scales = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.pre_encoded = pre_encoded
        self.max_images = max_images
        self.do_augmentation = do_augmentation

        # Mode: either train or val
        self.mode = mode

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

        # If a maximum number of images has been set
        # This is necessary for domain adaptation due to the difference in source-target dataset size
        if self.max_images:
            # Draw randomly #max_images form the image list
            choice = np.random.choice(len(self.image_list), self.max_images)
            # And select only those that have been drawn
            self.label_list = [self.label_list[c] for c in choice]
            self.image_list = [self.image_list[c] for c in choice]

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

        # Normalize((0.39068785, 0.40521392, 0.41434407), (0.29652068, 0.30514979, 0.30080369)),

    def __getitem__(self, index):
        # Read images and transform
        image = Image.open(self.image_list[index])

        # Labels are standard rgb images
        label = Image.open(self.label_list[index]).convert("RGB")

        if self.mode == 'train':
            trn = self.transform(image=np.array(image), mask=np.array(label))
            image, label = trn['image'], trn['mask']

        if self.mode == 'train' and self.do_augmentation:
            image = self.augment(image)

        # Encode label
        if self.loss == 'dice':
            # Encode label image
            label = encode_label_idda_dice(label).astype(np.uint8)
            label = torch.from_numpy(label)

        elif self.loss == 'crossentropy':
            label = encode_label_idda_crossentropy(label)
            label = torch.from_numpy(label)

        image = self.normalize(image).float()
        return image, label

    def __len__(self):
        return len(self.image_list)


def get_data_loaders(data, batch_size, num_workers, loss, pre_encoded, crop_height, crop_width, shuffle=True,
                     max_images=None, do_augmentation=True):
    """
    Build dataloader structures for train and validation

    Args:
        args: command line arguments

    Returns:
        dataloader structures for train and validation
    """

    # Build paths
    train_path = [os.path.join(data, 'train')]
    train_label_path = [os.path.join(data, 'train_labels')]

    test_path = os.path.join(data, 'val')
    test_label_path = os.path.join(data, 'val_labels')

    csv_path = os.path.join(data, 'class_dict.csv')
    # Train Dataloader
    dataset_train = IDDA(train_path, train_label_path, csv_path, (crop_height, crop_width), loss=loss,
                         pre_encoded=pre_encoded, max_images=max_images, do_augmentation=do_augmentation)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle
    )

    # Val Dataloader
    dataset_val = IDDA(test_path, test_label_path, csv_path, (crop_height, crop_width), loss=loss, mode='val',
                       pre_encoded=pre_encoded, do_augmentation=do_augmentation)

    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        num_workers=num_workers,
        shuffle=shuffle
    )

    return dataloader_train, dataloader_val
