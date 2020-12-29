# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import itertools
from typing import Any, Dict, List, Tuple, Union
import torch

import numpy as np
from PIL import Image, ImageDraw

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('agg')

from .boxes import Boxes

class Instances(object):
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same `__len__` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/Get a field:
       instances.gt_boxes = Boxes(...)
       print(instances.pred_masks)
       print('gt_masks' in instances)
    2. `len(instances)` returns the number of instances
    3. Indexing: `instances[indices]` will apply the indexing on all the fields
       and returns a new `Instances`.
       Typically, `indices` is a binary vector of length num_instances,
       or a vector of integer indices.
    """

    def __init__(self, image_size: Tuple[int, int], image_path: str = "", **kwargs: Any): #This Any is an class import from typing
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._image_path = image_path
        # self._fields: Dict[str, Any] = {}
        self._fields = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, value):
        print("setter called")
        self._image_path = value

    def __setattr__(self, name: str, val: Any) -> None: #Overwriting default python function
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        # data_len = len(value)
        # if len(self._fields):
        #     assert (
        #         len(self) == data_len
        #     ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Converts list fields to tensor
    def tensor(self):
        for k, v in self._fields.items():
            if isinstance(v, list):
                self._fields[k] = torch.tensor(v)

    #converts field to numpy
    def numpy(self):
        for k, v in self._fields.items():
            if isinstance(v, torch.Tensor):
                self._fields[k] = v.cpu().numpy()
            if isinstance(v, Boxes):
                self._fields[k] = v.tensor.cpu().numpy()

        return self

    def draw(self, direc, name=""):
        if self._image_path:
            img = plt.imread(self._image_path)
        else :
            img = np.ones(self.image_size, dtype=np.uint8)*255

        fig, ax = plt.subplots(1, figsize=(16,12))
        ax.imshow(img)

        color = {"pred_boxes":'b', "gt_boxes":'r', "proposal_boxes":'g'}

        for key in self._fields:
            if "box" in key:
                for box in self._fields[key]:
                    box = box.detach().cpu().numpy()
                    width= box[2]- box[0]
                    height = box[3]-box[1]
                    rect = patches.Rectangle(box[:2], width=width, height=height,
                        linewidth=1, fill=False, edgecolor=color[key])
                    ax.add_patch(rect)
           
            if "var" in key:
                for var, box in zip(self._fields[key], self._fields["pred_boxes"]):
                    var = var.detach().cpu().numpy()
                    box = box.detach().cpu().numpy()
                    sigma = np.sqrt(var)
                    ellip_top = patches.Ellipse([box[0], box[1]], width= 2*sigma[0], height= 2*sigma[1], fill = False, linewidth=2, edgecolor='b')
                    ellip_bottom = patches.Ellipse([box[2], box[3]], width= 2*sigma[2], height= 2*sigma[3], fill = False, linewidth=2, edgecolor='b')
                    ax.add_patch(ellip_top)
                    ax.add_patch(ellip_bottom)

        if self._image_path:
            plt.savefig(os.path.join(direc, name+self._image_path[-10:-3]+"png"))
        else:
            plt.savefig(os.path.join(direc, name+".png"))                

        plt.close()

    # Tensor-like methods
    def to(self, device: str) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size, self._image_path)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        ret = Instances(self._image_size, self._image_path)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            return len(v)
        raise NotImplementedError("Empty Instances does not support __len__!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_path={}, ".format(self._image_path)
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join(self._fields.keys()))
        return s

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "image_path={}, ".format(self._image_path)
        s += "fields=["
        for k, v in self._fields.items():
            s += "{} = {}, ".format(k, v)
        s += "])"
        return s
