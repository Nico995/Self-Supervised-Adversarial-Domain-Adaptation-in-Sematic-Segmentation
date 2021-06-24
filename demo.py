import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize

from dataset import camvid_data_loaders, idda_data_loaders
from model import BiSeNet
from utils import load_segm_args, reverse_one_hot, convert_class_to_color
# from model_old.build_BiSeNet import BiSeNet
from utils.utils import batch_to_plottable_image, encode_label_dice, get_label_info, encode_label_idda_dice
from PIL import Image

if __name__ == '__main__':

    # Read command line arguments
    args = load_segm_args()

    model = BiSeNet(args.num_classes, args.context_path).cuda()
    model.load_state_dict(torch.load(args.pretrained_model_path))

    #weighted crossentropy
    #hard negative mining
    #focal loss
    if args.dataset == 'CamVid':
        image_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/CamVid/test/Seq05VD_f01110.png'
        label_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/CamVid/test_labels/Seq05VD_f01110_L.png'
        normalize = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        label = torch.tensor(encode_label_dice(Image.open(label_path), get_label_info('data/CamVid/class_dict.csv')))

    else:
        image_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/IDDA/test/@277548.198@110606.177@Town10@ClearNoon@audi@1608476889@1.0000747884809016@1.0009874550785103@0.006818489637225866@373573@.jpg'
        label_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/IDDA/test_labels/@277548.198@110606.177@Town10@ClearNoon@audi@1608476889@1.0000747884809016@1.0009874550785103@0.006818489637225866@373573@.png'
        normalize = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        label = torch.tensor(encode_label_idda_dice(np.array(Image.open(label_path).convert('RGB'))))

    image = normalize(Image.open(image_path))
    model.eval()

    image, label = image.cuda(), torch.tensor(label.cuda())
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    predict = model(image.unsqueeze(0))
    ax[0].imshow(convert_class_to_color(reverse_one_hot(predict[0])))
    ax[0].set_title("predicted")

    ax[1].imshow(convert_class_to_color(reverse_one_hot(label)))
    ax[1].set_title("label")
    plt.show()

    exit()
