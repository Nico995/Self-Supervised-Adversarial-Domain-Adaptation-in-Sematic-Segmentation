import cv2
import numpy as np
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale, Compose


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
        color_jitter = ColorJitter(0.2 * self.s, 0.2 * self.s, 0.2 * self.s, 0.1 * self.s)
        rnd_color_jitter = RandomApply([color_jitter], p=0.8)
        rnd_gray = RandomGrayscale(p=0.2)
        color_distort = Compose([rnd_color_jitter, rnd_gray])
        a = color_distort(img)

        return a
