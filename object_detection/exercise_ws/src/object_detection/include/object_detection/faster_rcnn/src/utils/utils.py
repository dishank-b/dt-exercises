import torch
import os
import sys
import numpy as np
import math
import argparse
from PIL import Image, ImageDraw
import matplotlib.image as mpimg ## To load the image
from torch import optim
import os.path as path
from torchvision import transforms as T
from src.config import Cfg as cfg

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from .projection import ground_project
from .visualizer import Visualizer

def image_transform(cfg):
    """
    ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor  of shape (C x H x W) in the range [0.0, 1.0]

    """
    transform = T.Compose([T.ToTensor(),
                        T.Normalize(mean=cfg.BACKBONE.MEAN, std=cfg.BACKBONE.STD)])

    return transform

def toPIL(img: torch.Tensor):
    """
    T.ToPILImage() Converts an Tensor or numpy array in range [0, 1] with shape of (C x H x W) into an PIL image with range [0,255]
    """
    return T.Compose([T.Normalize( mean=[-mean/std for mean, std in zip(cfg.BACKBONE.MEAN, cfg.BACKBONE.STD)],
            std=[1.0/x for x in cfg.BACKBONE.STD]), T.ToPILImage()])(img)

def toNumpyImage(img: torch.Tensor):
    return T.Normalize( mean=[-mean/std for mean, std in zip(cfg.BACKBONE.MEAN, cfg.BACKBONE.STD)],
            std=[1.0/x for x in cfg.BACKBONE.STD])(img).mul(255).numpy().astype('uint8')

def disk_logger(images, direc, instances=None, rpn_proposals=None, image_paths=None):
    images = images.cpu()

    output_images = []

    for image, rpn_proposal, instance, path in zip(images, rpn_proposals, instances, image_paths):
        image = toPIL(image)
        
        img_visualizer = Visualizer(image, instance, path, rpn_proposal, cfg)

        if instance:
            img_visualizer.draw_instances()
            # img_visualizer.draw_projection()
            # img_visualizer.draw_proposals()
            # img_visualizer.draw_instance_prob()
        img_visualizer.save(direc)
        # img_visualizer.show()

        # print("{} written to disk".format(path[-10:]))
        output_images.append(img_visualizer.get_image())

    return output_images

def single_disk_logger(img, instances=None, rpn_proposals=None, image_path=None):
    img = img.cpu()
    img = toPIL(img)
    img_visualizer = Visualizer(img, instances, image_path, None, cfg)

    if instances:
        img_visualizer.draw_instances()
        img_visualizer.draw_projection()

    return img_visualizer.get_image()
        
def tb_logger(images, tb_writer, rpn_proposals=None, instances=None, name="Image"):

    image = toPIL(images[2].cpu())

    image_grid = image.resize((np.array(image.size)/1.5).astype(int))

    if rpn_proposals:
        proposal_locs = rpn_proposals[2].proposal_boxes[:50].tensor.cpu().numpy()
        proposal_img = draw_bbox(image.copy(), proposal_locs)

        image_grid = np.concatenate([image_grid, proposal_img], axis=1)

    if instances:
        pred = instances[2].pred_boxes.tensor.cpu().numpy()
        if instances[2].has("pred_variance"):
            sigma = instances[2].pred_variance.cpu().numpy()
        else:
            sigma=None
        img_cls = instances[2].pred_classes.cpu().numpy()

        prediction_img = draw_bbox(image.copy(), pred, img_cls, sigma)
        image_grid = np.concatenate([image_grid, prediction_img], axis=1)

    tb_writer.add_image(name, image_grid, dataformats='HWC')

