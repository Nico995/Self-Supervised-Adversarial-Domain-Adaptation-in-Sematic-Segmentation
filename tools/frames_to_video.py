import argparse
import glob
import os

import cv2


def main():
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/CamVid/', help='path of training data')
    parser.add_argument('--subfolder', type=str, default='test', help='')
    parser.add_argument('--video_code', type=str, default='Seq05VD', help='')

    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, args.subfolder, args.video_code + "_*.png")))

    img_array = []
    for image_file in files:
        img = cv2.imread(image_file)
        img_array.append(img)

    height, width, layers = cv2.imread(files[0]).shape
    size = (width, height)

    out = cv2.VideoWriter(f'../report_images/video/{args.video_code}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':
    main()
