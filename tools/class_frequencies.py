import argparse
from glob import glob
from os.path import join

import numpy as np
import tqdm
from PIL import Image
from matplotlib import pyplot as plt

colors = [
    [0, 0, 0],
    [70, 70, 70],
    [190, 153, 153],
    [72, 0, 90],
    [220, 20, 60],
    [153, 153, 153],
    [157, 234, 50],
    [128, 64, 128],
    [244, 35, 232],
    [107, 142, 35],
    [0, 0, 142],
    [102, 102, 156],
    [220, 220, 0],
    [250, 170, 30],
    [180, 165, 180],
    [111, 74, 0],
    [119, 11, 32],
    [0, 0, 230],
    [255, 0, 0],
    [152, 251, 152],
    [70, 130, 180],
    [230, 150, 140],
    [81, 0, 81],
    [0, 0, 0],
    [150, 100, 100],
    [45, 60, 150],
    [0, 0, 70]
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/IDDA/', help='path of training data')

    args = parser.parse_args()

    files = glob(join(args.data, "train_labels", "*.png"))
    classes = 27+1
    shape = (1920, 1080)

    running_counts = np.zeros(classes, dtype=np.int32)
    for i, file in enumerate(tqdm.tqdm(files)):

        # Open the image
        label = np.array(Image.open(file))

        # Map each RGBA color to the first byte (represents the class)
        label_class = label[:, :, 0]

        # Count the occurrences of every class
        counts = np.bincount(label_class.flatten(), minlength=classes).astype(np.int32)
        running_counts += counts

    class_map = [11, 1, 4, 11, 5, 3, 6, 6, 7, 10, 2, 11, 8, 11, 11, 11, 0, 11, 11, 11, 9, 11, 11, 11, 1, 11, 11]
    classes = ["unlabeled", "building", "fence", "other", "pedestrian", "pole", "roadline", "road", "sidewalk", "vegetation", "car", "wall", "tsign", "tlight", "guardrail", "dynamic", "bicycle", "motorcycle", "rider", "terrain", "sky", "railtrack", "ground", "statics", "bridge", "water", "truck"]    # used_classes = np.zeros(12)

    tot_pixels = len(files) * shape[0] * shape[1]
    running_counts[6] += running_counts[7]
    running_counts[7] = 0

    rel_freq = running_counts / tot_pixels
    alt_freq = 1 / (np.log(1.02 + (running_counts / tot_pixels)))

    for name, scaled, log_scaled in zip(classes, rel_freq, alt_freq):
        print(f"name: {name}\tscaled: {scaled:.5f}\tlog-scaled: {log_scaled:.5f}")


if __name__ == '__main__':
    main()
