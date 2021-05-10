import argparse
import glob
import os
import numpy as np
from PIL import Image
from utils import encode_label_dice, get_label_info
import tqdm


def main():
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/CamVid/', help='path of training data')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    parser.add_argument('--label_info', type=str, default='data/CamVid/class_dict.csv', help='metadata path')

    args = parser.parse_args()

    for label_dir in glob.glob(os.path.join(args.data, "*_labels")):
        labels_list = glob.glob(os.path.join(label_dir, "*.png"))
        # Progress bar
        tq = tqdm.tqdm(total=len(labels_list))
        tq.set_description(f'encoding {label_dir.split("/")[-1].split("_")[0]} labels')

        for label_file in labels_list:
            label = Image.open(label_file)
            label_prepared = encode_label_dice(label, get_label_info(args.label_info))
            np.save(label_file.split(".")[0], label_prepared)
            tq.update(1)

        tq.close()


if __name__ == '__main__':
    main()
