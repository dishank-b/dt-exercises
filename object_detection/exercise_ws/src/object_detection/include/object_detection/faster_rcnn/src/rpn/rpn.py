'''
Region proposal network baseclass.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import AnchorGenerator
from . import RPNProcessing

from ..utils import Boxes, Matcher, Box2BoxTransform
from ..nms import find_top_rpn_proposals

class RPNHead(nn.Module):
    """docstring for RPN"""

    def __init__(self, cfg, in_channels, num_anchors):
        super(RPNHead, self).__init__()

        ## Layer 1
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        ## classification layer
        self.classification_head = nn.Conv2d(in_channels, num_anchors, 1, 1, 0)
        
        ## Regression layer
        self.bbox_head = nn.Conv2d(in_channels, num_anchors*4, 1, 1, 0)

        ## Uncertainty layer
        self.uncertain_head = nn.Conv2d(in_channels, num_anchors*4, 1, 1, 0)

        # Wight Initialization
        for l in [self.conv1, self.bbox_head, self.classification_head, self.uncertain_head]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def freeze(self):
        # for name, module in self.named_children():
        #     # print(name)
        #     # freeze weights and biases of mentioned layers
        for p_name, p in self.named_parameters():
            print(p_name)
            p.requires_grad = False


    def forward(self, feature_map):

        x = F.relu(self.conv1(feature_map)) #x :[N, C=RPN.OUT_CHANNELS, H, W]

        bboxes_delta = self.bbox_head(x) # bboxes :[N, C=9*4, H, W]

        class_logits = self.classification_head(x) #class_logits :[N, C=9, H, W]

        return class_logits, bboxes_delta


class RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, in_channels):
        super(RPN, self).__init__()

        # fmt: off
        self.cfg = cfg
        self.min_box_side_len        = cfg.RPN.MIN_SIZE_PROPOSAL
        self.nms_thresh              = cfg.RPN.NMS_THRESH
        self.loss_weight             = cfg.RPN.LOSS_WEIGHT
        # fmt: on

        # Map from is_training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.RPN.POST_NMS_TOPK_TEST,
        }


        self.box2box_transform = Box2BoxTransform(weights=cfg.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(cfg.RPN.IOU_THRESHOLDS, cfg.RPN.IOU_LABELS, allow_low_quality_matches=True)
        self.anchors_generator = AnchorGenerator(cfg)
        self.rpn_head = RPNHead(cfg, in_channels, self.anchors_generator.num_anchors)

    def freeze(self):
        for name, module in self.named_children():
            try:
                module.freeze()			
            except:
                pass

    def forward(self, features, gt_target=None, image_sizes=None, is_training=True):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances] or None
            loss: dict[Tensor]
        """
        #  List[Boxes]: Length N
        gt_boxes = [x.gt_boxes for x in gt_target] if gt_target is not None else None

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        
        feature_shape = features.shape

        stride = round(image_sizes[-1]/feature_shape[-1])
        # print("Stride:", stride)

        #List[Boxes], return a list, each element is Box struct for each image in batch. Each Box struct is Tensor of all anchors in that image.
        # box struct: [HxWx9, 4]
        anchors = self.anchors_generator(feature_shape, stride)


        RPNProcessor = RPNProcessing(
            self.cfg,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            gt_boxes,
            self.anchor_matcher,
            self.box2box_transform,
            image_sizes
        )

        if is_training:
            losses = RPNProcessor.losses()
            {k: v * self.loss_weight for k, v in losses.items()}
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            image_sizes = [image_sizes]*feature_shape[0]
            proposals = find_top_rpn_proposals(
                RPNProcessor.predict_proposals(),
                RPNProcessor.predict_objectness_logits(),
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[is_training],
                self.post_nms_topk[is_training],
                self.min_box_side_len,
                is_training,
            )
            
            inds = [p.objectness_logits.sort(descending=True)[1] for p in proposals]
            proposals = [p[ind] for p, ind in zip(proposals, inds)]

        return proposals, losses
