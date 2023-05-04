import random
import glob
import os
import shutil

import argparse


def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-dr', '--input_path', default=None, type=str,
        help='path to the dataset'
    )

    parser.add_argument(
        '-d', '--output_path', default="splitted_data", type=str,
        help='output path to the splitted data'
    )

    args = vars(parser.parse_args())
    return args


def main(args):

    IN_DIR = args['input_path']
    OUT_DIR = args['output_path']

    # Validation split ratio.
    VALID_SPLIT = 0.17

    IMAGES_FOLDER = os.path.join(IN_DIR, 'images')
    LABELS_FOLDER = os.path.join(IN_DIR, 'annotations_yolo')

    TRAIN_IMAGES_DEST = os.path.join(OUT_DIR, 'train', 'images')
    TRAIN_LABELS_DEST = os.path.join(OUT_DIR, 'train', 'labels')
    VALID_IMAGES_DEST = os.path.join(OUT_DIR, 'valid', 'images')
    VALID_LABELS_DEST = os.path.join(OUT_DIR, 'valid', 'labels')

    os.makedirs(TRAIN_IMAGES_DEST, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DEST, exist_ok=True)
    os.makedirs(VALID_IMAGES_DEST, exist_ok=True)
    os.makedirs(VALID_LABELS_DEST, exist_ok=True)

    all_src_images = sorted(os.listdir(IMAGES_FOLDER))
    all_src_labels = sorted(os.listdir(LABELS_FOLDER))

    # Randomoze images and annotations list in same order.
    temp = list(zip(all_src_images, all_src_labels))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    temp_images, temp_labels = list(res1), list(res2)

    print(temp_images[:3])
    print(temp_labels[:3])

    num_training_images = int(len(temp_images)*(1-VALID_SPLIT))
    num_valid_images = int(len(temp_images)-num_training_images)

    print(num_training_images, num_valid_images)

    train_images = temp_images[:num_training_images]
    train_labels = temp_labels[:num_training_images]

    valid_images = temp_images[num_training_images:len(all_src_images)]
    valid_labels = temp_labels[num_training_images:len(all_src_images)]

    print(train_images[:3])
    print(valid_images[:3])

    for i in range(len(train_images)):
        shutil.copy(
            os.path.join(IMAGES_FOLDER, train_images[i]),
            os.path.join(TRAIN_IMAGES_DEST, train_images[i])
        )
        shutil.copy(
            os.path.join(LABELS_FOLDER, train_labels[i]),
            os.path.join(TRAIN_LABELS_DEST, train_labels[i])
        )

    for i in range(len(valid_images)):
        shutil.copy(
            os.path.join(IMAGES_FOLDER, valid_images[i]),
            os.path.join(VALID_IMAGES_DEST, valid_images[i])
        )
        shutil.copy(
            os.path.join(LABELS_FOLDER, valid_labels[i]),
            os.path.join(VALID_LABELS_DEST, valid_labels[i])
        )

    print("Success....")


if __name__ == '__main__':
    args = parse_opt()
    main(args)
