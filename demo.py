import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import get_data_loaders
from model import BiSeNet
from utils import load_args, reverse_one_hot, convert_class_to_color

if __name__ == '__main__':

    # Reproducibility
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(42)
    # python seed
    random.seed(42)
    # seed the global NumPy RNG
    np.random.seed(42)

    # Read command line arguments
    args = load_args()

    # Get dataloader structures

    _, dataloader_val = get_data_loaders(args)

    # build model
    model = BiSeNet(args.num_classes, args.context_path).cuda()
    model.load_state_dict(torch.load("checkpoints_18_sgd/latest_dice_loss.pth"))
    model.eval()

    print(len(dataloader_val))
    for i, (data, label) in enumerate(dataloader_val):
        label, data = label.cuda(), data.cuda()
        predict = model(data)
        predict_image = convert_class_to_color(reverse_one_hot(predict[0]))
        plt.imshow(predict_image)
        plt.title("predicted1")
        plt.show()

        label_image = convert_class_to_color(reverse_one_hot(label[0]))
        plt.imshow(label_image)
        plt.title("label")
        plt.show()
        exit()
