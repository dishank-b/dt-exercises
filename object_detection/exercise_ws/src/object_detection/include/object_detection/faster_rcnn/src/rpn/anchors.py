'''
Base class to generate anchors. 
This code will mostly be used for training!(You never know though!)
'''
import copy
import numpy as np
import torch
from torch import nn
from ..utils import Boxes
import math
from PIL import Image, ImageDraw


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())



class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, device=torch.device('cuda')):
        super(AnchorGenerator, self).__init__()
        sizes         = cfg.ANCHORS.ANCHOR_SCALES
        aspect_ratios = cfg.ANCHORS.ASPECT_RATIOS

        self._num_anchors_per_pixel = len(sizes)*len(aspect_ratios)
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        base_anchors = self.generate_base_anchors(sizes, aspect_ratios, device)
        self.register_buffer("base_anchors", base_anchors)

    def _create_grid_offsets(self, size, stride, device):
        grid_height, grid_width = size
        shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(
            0, grid_height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        
        return shift_x, shift_y

    def grid_anchors(self, grid_sizes, stride):
        shift_x, shift_y = self._create_grid_offsets(grid_sizes, stride, self.base_anchors.device)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        anchors = shifts.view(-1, 1, 4) + self.base_anchors.view(1, -1, 4) # anchors: [H*W, num_of_achors_per_place=9, 4]
        anchors = anchors.reshape(-1, 4) # anchors: [tot_num_anchors=H*W*9, 4] : first 9 boxes are anchors for first (x,y) and so on. so

        # nup_arr = np.ones((375,1242,3), dtype='uint8')*255
        # img = Image.fromarray(nup_arr)
        # drawer = ImageDraw.Draw(img)
        # for i in anchors[np.random.random_integers(0, 16734, 50)]:
        #     drawer.rectangle(i.cpu().numpy() ,outline='red')
        # img.save("/network/home/bansaldi/Denso-OD/logs/both_stage/results/some.jpg", 'JPEG')

        return anchors

    def generate_base_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2), device=torch.device('cuda')):
        """
        Generate a tensor storing anchor boxes, which are continuous geometric rectangles
        centered on one feature map point sample. We can later build the set of anchors
        for the entire feature map by tiling these tensors; see `meth:grid_anchors`.

        Args:
            sizes (tuple[float]): Absolute size of the anchors in the units of the input
                image (the input received by the network, after undergoing necessary scaling).
                The absolute size is given as the side length of a box.
            aspect_ratios (tuple[float]]): Aspect ratios of the boxes computed as box
                height / width.

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])

        return torch.tensor(anchors, device=device)

    def forward(self, features_shape, stride):
        """
        Args:
            features (Tensor): Tensor of feature map on which to generate anchors.

        Returns:
            list[Boxes]: a list of #image elements.
        """
        num_images = features_shape[0]
        grid_sizes = features_shape[-2:]
        anchors = self.grid_anchors(grid_sizes, stride)
        anchors_image = Boxes(anchors) 
        anchors_batch = [copy.deepcopy(anchors_image) for _ in range(num_images)]
    
        # List(Boxes) len(anchor_batch) = batch_size, 
        # each element is a Box Strucutre which is list of all anchor in image i. 
        # Box = [HxWx9, 4]
        return anchors_batch 

    @property
    def num_anchors(self):
        return self._num_anchors_per_pixel


