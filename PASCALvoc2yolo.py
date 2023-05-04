import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join

import argparse
import yaml


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, xmax, ymin, ymax
    x_center = ((bbox[1] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[2]) / 2) / h
    width = (bbox[1] - bbox[0]) / w
    height = (bbox[3] - bbox[2]) / h
    return [x_center, y_center, width, height]


def Convert2yolo(label_input_dir, image_dir, output_dir, classes):

    files = glob.glob(os.path.join(label_input_dir, '*.xml'))
    for fil in files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        if not os.path.exists(os.path.join(image_dir, f"{filename}.png")):
            print(f"{filename} image does not exist!")
            continue

        result = []

        # parse the content of the xml file
        tree = ET.parse(fil)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall('object'):
            label = obj.find("name").text
            # check for new classes and append to list
            if label not in classes:
                classes.append(label)
            index = classes.index(label)
            pil_bbox = [int(x.text) for x in obj.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")

        if result:
            # generate a YOLO format text file for each xml file
            with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))


def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-dr', '--input_path', default=None, type=str,
        help='path to the dataset in PASCAL VOC format'
    )

    parser.add_argument(
        '-d', '--output_path', default="annotations_yolo", type=str,
        help='path to the annotations in the yolo format'
    )

    parser.add_argument(
        '-c', '--config', default=None,
        help='path to the data config file'
    )

    args = vars(parser.parse_args())
    return args


def main(args):

    # Load the data configurations
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)

    CLASSES = data_configs['names']

    print(" we assume that the folder containing the images is called 'images' and the folder containing the annotation is called 'annotations' ")
    # Settings/parameters/constants.
    DIR_IMAGES = args['input_path']+"/images"
    DIR_LABELS = args['input_path']+"/annotations"

    print("Started processing")
    OUT_DIR = args['output_path']

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    Convert2yolo(DIR_LABELS, DIR_IMAGES, OUT_DIR, CLASSES)

    print("Finished processing")


if __name__ == '__main__':
    args = parse_opt()
    main(args)
