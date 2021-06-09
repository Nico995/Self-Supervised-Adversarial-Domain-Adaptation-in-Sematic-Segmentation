from abc import ABC

import cv2
import random
import numpy as np
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale, Compose
from albumentations import DualTransform, to_tuple
import albumentations.augmentations.functional as F

class RandomGaussianBlur(object):
    """
    Inspired by SwAV code on github (https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py)
    """

    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if do_it:
            sigma = np.random.rand() * 1.9 + 0.1
            img = cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)

        return img


class ColorDistortion(object):
    """
    Inspired by SwAV code on github (https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py)
    """

    def __init__(self, strength=1.0):
        self.s = strength

    def __call__(self, img):
        color_jitter = ColorJitter(0.4 * self.s, 0.4 * self.s, 0.4 * self.s, 0.1 * self.s)
        rnd_color_jitter = RandomApply([color_jitter], p=0.8)
        rnd_gray = RandomGrayscale(p=0.1)
        color_distort = Compose([rnd_color_jitter, rnd_gray])
        a = color_distort(img)

        return a


class RandomDiscreteScale(DualTransform):
    def __init__(self, scales, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5):
        super(RandomDiscreteScale, self).__init__(always_apply, p)
        self.scales = scales
        self.interpolation = interpolation

    def get_params(self):
        return {"scale": random.choice(self.scales)}

    def apply(self, img, scale=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.scale(img, scale, interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, scale=0, **params):
        return F.keypoint_scale(keypoint, scale, scale)

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "scale_limit": to_tuple(self.scale_limit, bias=-1.0)}