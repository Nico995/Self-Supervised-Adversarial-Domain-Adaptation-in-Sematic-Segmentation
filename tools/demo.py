import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

from models import BiSeNet
from utils import load_segm_args, reverse_one_hot, convert_class_to_color
# from model_old.build_BiSeNet import BiSeNet
from utils.utils import encode_label_dice, get_label_info, encode_label_idda_dice

if __name__ == '__main__':

    savedir = '/home/nicola/Documents/uni/MLDL/project/BiSeNet/report_images/inference'

    # Read command line arguments
    args = load_segm_args()

    model = BiSeNet(args.num_classes, args.context_path).cuda()
    model.load_state_dict(torch.load(args.pretrained_model_path))

    # weighted crossentropy
    # hard negative mining
    # focal loss
    if args.dataset == 'CamVid':
        image_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet//data/CamVid/test/Seq05VD_f04530.png'
        label_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet//data/CamVid/test_labels/Seq05VD_f04530_L.png'
        normalize = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        label = torch.tensor(encode_label_dice(Image.open(label_path), get_label_info('/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/CamVid/class_dict.csv')))

    else:
        image_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/IDDA/test/@277523.219@110462.794@Town10@ClearNoon@audi@1608287362@0.9987782228147495@1.0007638637244156@1.0836338996887207@204741@.jpg'
        label_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/IDDA/test_labels/@277523.219@110462.794@Town10@ClearNoon@audi@1608287362@0.9987782228147495@1.0007638637244156@1.0836338996887207@204741@.png'
        normalize = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        label = torch.tensor(encode_label_idda_dice(np.array(Image.open(label_path).convert('RGB'))))

    image = normalize(Image.open(image_path))
    model.eval()

    image, label = image.cuda(), torch.tensor(label.cuda())

    fig = plt.figure()
    predict = model(image.unsqueeze(0))
    plt.imshow(convert_class_to_color(reverse_one_hot(predict[0])))
    plt.axis('off')
    plt.tight_layout()
    loss_version = args.pretrained_model_path.split('/')[-2].split('-')[-1]
    plt.savefig(os.path.join(savedir, args.dataset + '_' + loss_version), bbox_inches='tight')
    print('fig saved at ', os.path.join(savedir, args.dataset + '_' + loss_version))
    exit()
