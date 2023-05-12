"""
USAGE

# Generate Transfer Adversarial Attack:
python pgd.py -d <dir_image> -dest <dir-result>

"""
import os
import argparse

import numpy as np
import yaml
import cv2
import tqdm
import torch

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.object_detection import PyTorchFasterRCNN


def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--images-dir', default=None, type=str, dest='images-path',
        help='path to the images folder to perform the attack upon'
    )

    parser.add_argument(
        '-dest', '--dest-dir',
        default="inference/", type=str, dest='result-dir',
        help='path to the destination folder'
    )

    args = vars(parser.parse_args())

    return args


def main(args):
    # get the training test path, Number of classes and the classes from the config file
    with open('/home/aya/Desktop/Kitti_FasterRCNN/data_configs/data.yaml') as file:
        data_configs = yaml.safe_load(file)

    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    SAVE_VALID_PREDICTIONS = data_configs['SAVE_VALID_PREDICTION_IMAGES']

    TEST_DIR_IMAGES = args['images-path']
    DEST_DIR = args['result-dir']
    os.makedirs(DEST_DIR, exist_ok=True)  # make the destination dir

    # Create ART object detector
    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    )

    # Create and run attack
    eps = 64
    attack = ProjectedGradientDescent(
        estimator=frcnn, eps=eps, eps_step=2, max_iter=10)

    for i in os.listdir(TEST_DIR_IMAGES):

        img = cv2.imread(os.path.join(TEST_DIR_IMAGES, i))

        img = cv2.resize(img, dsize=(
            img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        image = np.stack([img], axis=0).astype(np.float32)

        # apply the attack
        image_adv = attack.generate(x=image, y=None)
        img_res = image_adv[0].astype(np.uint8)

        cv2.imwrite(DEST_DIR+"/"+i, img_res)

    if __name__ == '__main__':
        args = parse_opt()
        main(args)
