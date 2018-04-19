"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import numpy as np

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import scipy.misc
import skimage.color
import skimage.io
import xmltodict

import zipfile
import urllib.request
import shutil

from config import Config
from coco import CocoConfig
import utils
import model_detrac as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################


class DetracConfig(CocoConfig):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "detrac"


############################################################
#  Dataset
############################################################
class DetracDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_detrac(self, height=540, width=960, mode=None):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        self.xml_annotation_path = "/Users/tingyumao/Documents/experiments/traffic/detrac/data/DETRAC-Train-Annotations-XML"
        self.image_path = "/Users/tingyumao/Documents/experiments/traffic/detrac/data/Insight-MVT_Annotation_Train"
        self.detect_annotation_path = "./data/detrac/annotation"
        self.detect_ignore_path = "./data/detrac/ignore"

        if not os.path.isdir(self.detect_annotation_path) or not os.path.isdir(self.detect_ignore_path):
            self.translate_xml()

        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

        # Add classes
        for i, c in enumerate(class_names):
            if c == "BG":
                continue
            self.add_class("detrac", i, c)

        # Add image by adding image basic info like image id, source (here it is "detrac"), path.
        img_cnt = 0
        if mode == "train":
            seq_info = sorted([x for x in os.listdir(self.image_path) if x.startswith("MVI_")])[:-10]
        elif mode == "val":
            seq_info = sorted([x for x in os.listdir(self.image_path) if x.startswith("MVI_")])[-10:]
        else:
            IOError("mode should be either train or val")
        for seq in seq_info:
            images = sorted([x for x in os.listdir(os.path.join(self.image_path, seq)) if x.endswith(".jpg")])
            for img in images:
                frame_id = int(img.replace(".jpg", "").replace("img", ""))
                if os.path.isfile(os.path.join(self.detect_annotation_path, seq, str(frame_id).zfill(5) + ".txt")):
                    self.add_image("detrac", image_id=img_cnt, path=os.path.join(self.image_path, seq, img),
                                   seq_id=seq, frame_id=frame_id, height=height, width=width)
                    img_cnt += 1

        print("number of total image is ", img_cnt)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        seq_id = info["seq_id"]

        # read raw image
        image = skimage.io.imread(info['path'])

        # read ignored region and set ignore region to zero
        with open(os.path.join(self.detect_ignore_path, seq_id + ".txt"), "r") as f:
            for line in f:
                x1, y1, x2, y2 = line.replace("\n", "").split(" ")
                x1, y1, x2, y2 = [int(float(x)) for x in [x1, y1, x2, y2]]
                image[y1:y2, x1:x2] = 0

        return image.astype("uint8")

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        pass

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        mask: h x w x num_instance
        """
        info = self.image_info[image_id]
        seq_id = info["seq_id"]
        frame_id = info["frame_id"]
        height = info["height"]
        width = info["width"]
        class_names = self.class_names

        # read txt annotation file
        mask = []
        class_list = []
        with open(os.path.join(self.detect_annotation_path, seq_id, str(frame_id).zfill(5) + ".txt"), "r") as f:
            for line in f:
                car_coco_class, x1, y1, x2, y2 = line.replace("\n", "").split(" ")
                x1, y1, x2, y2 = [int(float(x)) for x in [x1, y1, x2, y2]]
                instance_mask = np.zeros((height, width)).astype("uint8")
                instance_mask[y1:y2, x1:x2] = 1
                mask.append(instance_mask)
                class_list.append(car_coco_class)

        # Convert mask into numpy array [h, w, num_instance]
        mask = np.stack(mask, 2).astype("uint8")
        # Map class names to class IDs.
        class_ids = np.array([class_names.index(c) for c in class_list])

        return mask, class_ids.astype(np.int32)

    @staticmethod
    def read_bbox(annotations):
        if isinstance(annotations, dict):
            annotations = [annotations]
        # generate bbox for a specific frame
        bbox = list()
        # occlusion_bbox = list()
        for box in annotations:
            car_id = box["@id"]
            x1, y1, w, h = float(box["box"]["@left"]), float(box["box"]["@top"]), float(box["box"]["@width"]), float(
                box["box"]["@height"])
            x2, y2 = x1 + w, y1 + h
            car_type = box["attribute"]["@vehicle_type"]
            # ignore bounding boxes which are almost blocked by other objects.
            ov_rate = 0
            if "occlusion" in box.keys():
                occlusion = box["occlusion"]["region_overlap"]
                if isinstance(occlusion, list):
                    for o in occlusion:
                        ox1, oy1, ow, oh = float(o["@left"]), float(o["@top"]), float(o["@width"]), float(o["@height"])
                        ov_rate += (ow * oh) / (w * h)
                else:
                    ox1, oy1, ow, oh = float(occlusion["@left"]), float(occlusion["@top"]), float(
                        occlusion["@width"]), float(occlusion["@height"])
                    ov_rate = (ow * oh) / (w * h)
            if ov_rate < 0.9:
                bbox.append([car_id, car_type, x1, y1, x2, y2])

        return bbox

    def translate_xml(self):
        # create the saving directory
        if not os.path.isdir(self.detect_annotation_path):
            os.makedirs(self.detect_annotation_path)
        if not os.path.isdir(self.detect_ignore_path):
            os.makedirs(self.detect_ignore_path)
        # read xml annotation file
        seq_info = sorted([x for x in os.listdir(self.image_path) if x.startswith("MVI_")])
        for seq in seq_info:
            # 0. read and parse xml file
            xml_file_path = os.path.join(self.xml_annotation_path, seq + ".xml")
            with open(xml_file_path) as f:
                annotations = xmltodict.parse(f.read())
            # 1. read and save ignored region
            with open(os.path.join(self.detect_ignore_path, seq + ".txt"), "w+") as f:
                if annotations["sequence"]["ignored_region"] is not None:
                    if isinstance(annotations["sequence"]["ignored_region"]["box"], dict):
                        annotations["sequence"]["ignored_region"]["box"] = [
                            annotations["sequence"]["ignored_region"]["box"]]
                    for box in annotations["sequence"]["ignored_region"]["box"]:
                        x1, y1, w, h = float(box["@left"]), float(box["@top"]), float(box["@width"]), float(
                            box["@height"])
                        x2, y2 = x1 + w, y1 + h
                        string = ' '.join([str(x) for x in [x1, y1, x2, y2]])
                        f.write(string + "\n")
            # 2. read and save annotations
            if not os.path.isdir(os.path.join(self.detect_annotation_path, seq)):
                os.makedirs(os.path.join(self.detect_annotation_path, seq))
            frame_annotations = annotations["sequence"]["frame"]
            for _, bbox_annotations in enumerate(frame_annotations):
                frame_id = bbox_annotations['@num']
                bbox = self.read_bbox(bbox_annotations["target_list"]["target"])
                # only add bbox into all_bbox for detection model
                with open(os.path.join(self.detect_annotation_path, seq, frame_id.zfill(5) + ".txt"), "w+") as f:
                    for box in bbox:
                        car_id, car_type, x1, y1, x2, y2 = box
                        if car_type in ["car", "other", "van"]:
                            car_coco_class = "car"
                        else:
                            car_coco_class = "bus"
                        string = ' '.join([str(x) for x in [car_coco_class, x1, y1, x2, y2]])
                        f.write(string + "\n")


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    """
    python detrac.py train --dataset=/path/to/detrac/ --model=coco
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on UA-Detrac.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' UA-Detrac")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/detrac/",
                        help='Directory of the UA-Detrac dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DetracConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset
        dataset_train = DetracDataset()
        dataset_train.load_detrac(mode="train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = DetracDataset()
        dataset_val.load_detrac(mode="val")
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        # print("Fine tune Resnet stage 4 and up")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=120,
        #             layers='4+')
        #
        # # Training - Stage 3
        # # Fine tune all layers
        # print("Fine tune all layers")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10,
        #             epochs=160,
        #             layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        coco = dataset_val.load_coco(args.dataset, "minival", year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
