#!/usr/bin/env python3

import json
import os
import cv2
import numpy as np

from utils import makedirs

DATASET_PATH="../../dataset"

with open(f"{DATASET_PATH}/real_data/duckietown object detection dataset/annotation/final_anns.json") as anns:
    annotations = json.load(anns)

npz_index = 0
while os.path.exists(f"{DATASET_PATH}/{npz_index}.npz"):
    npz_index += 1

def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(f"{DATASET_PATH}"):
        np.savez(f"{DATASET_PATH}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1

for filename in os.listdir(f"{DATASET_PATH}/real_data/duckietown object detection dataset/frames"):
    img = cv2.imread(f"{DATASET_PATH}/real_data/duckietown object detection dataset/frames/{filename}")

    boxes = []
    classes = []

    if filename not in annotations:
        continue

    for detection in annotations[filename]:
        box = detection["bbox"]
        label = detection["cat_name"]

        if label not in ["duckie", "cone"]:
            continue

        orig_x_min, orig_y_min, orig_w, orig_h = box
       
        boxes.append([orig_x_min, orig_y_min, orig_x_min+orig_w, orig_y_min+orig_h])
        classes.append(0 if label == "duckie" else 1)

    if len(boxes) == 0:
        continue

    save_npz(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        np.array(boxes),
        np.array(classes)
    )
