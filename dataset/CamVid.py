import glob
import os
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, RandomResizedCrop

from utils import one_hot_it_v11, one_hot_it_v11_dice, get_label_info


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
        self.image_trans = Compose([
            RandomResizedCrop(self.image_size, (0.5, 2)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.label_trans = Compose([
            RandomResizedCrop(self.image_size, (0.5, 2)),
        ])

    def __getitem__(self, index):
        # Seed transformations (have to be the same random crop for image and label)
        seed = random.random()
        torch.random.manual_seed(int(seed))
        random.seed(seed)

        # Read images and transform
        image = Image.open(self.image_list[index])
        image = self.image_trans(image).float()

        # Read labels and transform
        label = Image.open(self.label_list[index])
        label = np.array(self.label_trans(label))

        if self.loss == 'dice':
            # label -> [num_classes, H, W]
            label = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label)

            return image, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label).long()

            return image, label

    def __len__(self):
        return len(self.image_list)
