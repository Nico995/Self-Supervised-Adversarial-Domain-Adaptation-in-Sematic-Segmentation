import cv2
import numpy as np
from flask import Flask, render_template, Response
from time import sleep
import time
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.backends import cudnn
from models import BiSeNet
from utils import load_segm_args, reverse_one_hot, convert_class_to_color
# from model_old.build_BiSeNet import BiSeNet
from utils.utils import encode_label_dice, get_label_info, encode_label_idda_dice, convert_class_to_color_V2


normalize = Compose([
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class VideoCamera(object):
    def __init__(self, video_file):
        self.video_file = video_file
        self.start_feed()
        self.video_out = cv2.VideoWriter(f'../report_images/video/segmented.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (960, 720))

    def start_feed(self):
        self.video = cv2.VideoCapture(self.video_file)

    def __del__(self):
        self.video.release()

    def get_frame(self):

        start = time.time_ns() // 1000000

        success, image = self.video.read()
        if not success:
            self.video_out.release()
            print("video finished")
            exit()

        gpu_image = normalize(image).cuda().unsqueeze(0)
        predict = model(gpu_image)[0]

        predict = reverse_one_hot(predict)

        # This slows down of about 6 fps
        predict = convert_class_to_color_V2(predict).detach().cpu().numpy().astype(np.uint8)
        predict = cv2.cvtColor(predict, cv2.COLOR_BGR2RGB)
        end = time.time_ns() // 1000000

        cv2.putText(predict, f'FPS {1000//(end-start)}', (900, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 51), 2)
        out = np.concatenate((cv2.resize(image, (480, 360)), cv2.resize(predict, (480, 360))), axis=1)
        ret, jpeg = cv2.imencode('.jpg', out)
        self.video_out.write(out)

        return jpeg.tobytes()


app = Flask(__name__)


def gen_frames(camera):

    while True:

        frame = camera.get_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Video streaming home page."""
    # return Response(gen_frames(VideoCamera('../report_images/video/Seq05VD.mp4')), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(gen_frames(VideoCamera('../report_images/video/0005VD.mp4')), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':

    print('Loading pretrained model')
    model = BiSeNet(12, 'resnet101').cuda()
    model.load_state_dict(torch.load('../runs_data/da-res101-crossentropy/best_loss.pth'))
    model.eval()
    cudnn.benchmark = True

    print('Done loading')

    app.run(debug=False)