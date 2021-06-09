import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize

from dataset import camvid_data_loaders, idda_data_loaders
from model import BiSeNet
from utils import load_segm_args, reverse_one_hot, convert_class_to_color
# from model_old.build_BiSeNet import BiSeNet
from utils.utils import batch_to_plottable_image
from PIL import Image
if __name__ == '__main__':

    # Reproducibility
    # # seed the RNG for all devices (both CPU and CUDA)
    # torch.manual_seed(seed)
    # # python seed
    # random.seed(seed)
    # # seed the global NumPy RNG
    # np.random.seed(seed)

    # Read command line arguments
    args = load_segm_args()

    # Get dataloader structures

    if args.dataset == 'IDDA':
        _, dataloader_val = idda_data_loaders(args.data, 1, 1, args.loss, args.pre_encoded, args.crop_height, args.crop_width, shuffle=True)
    else:
        image = Image.open('/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/CamVid/test/0001TP_008550.png')
        plt.imshow(np.array(image))
        plt.show()
        label = Image.open('/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/CamVid/test_labels/0001TP_008550_L.png')
        image = Compose([
            ToTensor(),
            Normalize((0.39068785, 0.40521392, 0.41434407), (0.29652068, 0.30514979, 0.30080369))])(image).unsqueeze(0)
            # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # build model
    model = BiSeNet(args.num_classes, args.context_path)
    model.load_state_dict(torch.load(args.pretrained_model_path))
    model.eval()

    plt.imshow(batch_to_plottable_image(image))
    plt.show()
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()

    predict = model(image)
    predict_image = convert_class_to_color(reverse_one_hot(predict[0]))
    ax[0].imshow(predict_image)
    ax[0].set_title("predicted")

    if args.dataset == 'IDDA':
        label = label[0]
        label_image = convert_class_to_color(reverse_one_hot(label))
        # label_image = np.transpose(label_image, (1, 0, 2))
    else:
        pass
        # label_image = convert_class_to_color(reverse_one_hot(label[0]))

    ax[1].imshow(label)
    ax[1].set_title("label")
    plt.show()
    exit()
