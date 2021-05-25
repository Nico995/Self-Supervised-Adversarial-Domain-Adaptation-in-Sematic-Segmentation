import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import camvid_data_loaders
from model import BiSeNet
from utils import load_args, reverse_one_hot, convert_class_to_color

if __name__ == '__main__':

    seed = 41
    # Reproducibility
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)
    # python seed
    random.seed(seed)
    # seed the global NumPy RNG
    np.random.seed(seed)

    # Read command line arguments
    args = load_args()

    # Get dataloader structures

    _, dataloader_val = camvid_data_loaders(args, shuffle=True)

    # build model
    model = BiSeNet(args.num_classes, args.context_path).cuda()
    model.load_state_dict(torch.load(args.pretrained_model_path))
    model.eval()

    print(len(dataloader_val))
    for i, (data, label) in enumerate(dataloader_val):
        fig, ax = plt.subplots(1, 2)
        ax = ax.ravel()
        label, data = label.cuda(), data.cuda()
        predict = model(data)
        predict_image = convert_class_to_color(reverse_one_hot(predict[0]))
        ax[0].imshow(predict_image)
        ax[0].set_title("predicted")

        label_image = convert_class_to_color(reverse_one_hot(label[0]))
        ax[1].imshow(label_image)
        ax[1].set_title("label")
        plt.show()
        exit()
