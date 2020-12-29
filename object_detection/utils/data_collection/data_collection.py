#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from skimage.measure import label, regionprops
from PIL import Image

from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask


DATASET_DIR="../../dataset/sim_data"

# Duckie, Cone, Truck, Bus
colors = [[100, 117, 226],[226, 111, 101],[116,114,117],[216,171, 15]]

npz_index = 0
def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(DATASET_DIR):
        img_save = Image.fromarray(img)
        img_save.save(f"{DATASET_DIR}/{npz_index}.png")
        f = open(f"{DATASET_DIR}/{npz_index}.txt", "w")
        for box, clss in zip(boxes, classes):
            print(f"{clss} {box[0]} {box[1]} {box[2]} {box[3]}", file=f)
        f.close()
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1

def clean_segmented_image(seg_img):
    kernel = np.ones((5,5),np.uint8)
    boxes = []
    classes = []
    for idx, color in enumerate(colors):
        mask = (seg_img==color).all(axis=2)
        mask.dtype = np.uint8
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        fig,ax = plt.subplots(1)
        ax.imshow(np.expand_dims(mask, axis=2))
        
        for prop in regionprops(label(mask)):
            if prop.area>50:
                # left, top, right, bottom in image frame. Sasically box[0] and box[2] is along axis 1 i.e column 
                box = np.array(prop.bbox)[[1,0,3,2]]
                boxes.append(box)
                classes.append(idx)
                # rect = patches.Rectangle(box[:2],box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
                # ax.add_patch(rect)
        # plt.show()
    return np.array(boxes), np.array(classes)

seed(123)
environment = launch_env(randomize_maps_on_reset=True)

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

TOTAL_IMAGES = 2000

while npz_index<TOTAL_IMAGES:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        if nb_of_steps%10==0:
            boxes, classes = clean_segmented_image(segmented_obs)
            if len(boxes)>0:
                save_npz(obs, boxes, classes)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break