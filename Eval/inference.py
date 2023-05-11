'''
USAGE
This code performs predictions and out puts a folder containing prediction annotation
the format is <object-name> <score> <left> <top> <right> <bottom>

Generate predictions by :
python inference.py -d <path to the images folder> -dest <path to the results folder>
'''

import os
import torch
from ultralytics import YOLO

import argparse
from PIL import Image
import numpy as np
import cv2


def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--images-dir', default=None, type=str, dest='images-path',
        help='path to the images folder to perform prediction upon'
    )

    parser.add_argument(
        '-dest', '--dest-dir',
        default="inference/", type=str, dest='result-dir',
        help='path to the destination folder'
    )

    args = vars(parser.parse_args())
    return args


def main(args):

    images_path = args['images-path']
    path = args['result-dir']

    NUM_CLASSES = 9
    KITTI_INSTANCE_CATEGORY_NAMES = {0: u'Cyclist', 1: u'DontCare', 2: u'Misc', 3: u'Person_sitting', 4: u'Tram', 5: u'Truck', 6: 'Van', 7: u'car', 8: u'person'
                                     }

    DEVICE = torch.device('cuda')

    model = YOLO("../best.pt")

    print("pretrained model imported....")
    os.makedirs(path, exist_ok=True)  # make the destination dir

    print("prediction...")
    for i in os.listdir(images_path):

        img = Image.open(images_path+i)
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        results = model.predict(img, verbose=False)

        # seperate the 3 keys (boxes, labels, scores)
        # bboxes, labels, scores = results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf #same results
        bboxes, labels, scores = results[0].boxes.boxes, results[0].boxes.cls, results[0].boxes.conf
        OBJECT = ""

        for j in range(len(labels)):
            OBJECT += KITTI_INSTANCE_CATEGORY_NAMES[int(labels[j])]
            OBJECT += " "
            OBJECT += str(float(scores[j]))
            OBJECT += " "
            OBJECT += str(int(bboxes[j][0]))
            OBJECT += " "
            OBJECT += str(int(bboxes[j][1]))
            OBJECT += " "
            OBJECT += str(int(bboxes[j][2]))
            OBJECT += " "
            OBJECT += str(int(bboxes[j][3]))
            OBJECT += "\n"

        fileName = i.split(".")
        fileName = fileName[0]+".txt"
        with open(os.path.join(path+fileName), 'w') as f:
            f.write(OBJECT)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
