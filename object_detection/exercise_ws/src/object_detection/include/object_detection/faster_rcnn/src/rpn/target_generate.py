
'''
Class to generate targets for 
Region proposal network. 
'''

import torch
import sys
import numpy as np
from . import AnchorGenerator
from ..utils import Boxes, pairwise_iou, subsample_labels
from ..loss import rpn_losses

class RPNProcessing(object):

    def __init__(
        self,
        cfg,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,
        gt_boxes,
        matcher,
        box2box_transform,
        image_sizes=None
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.cfg = cfg
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas
        self.anchors = anchors #[N, num_of_anchor, 4] - These will be boxes class
        self.gt_boxes = gt_boxes # [batch_size, M, 4] M= number of gt boxes in an image - These will be Boxes class
        self.image_size = image_sizes
        # self.num_images = len(gt_boxes)
        
        self.box2box_transform = box2box_transform
        self.anchor_matcher = matcher
        
        self.num_images = cfg.TRAIN.BATCH_SIZE
        self.batch_size_per_image = cfg.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction  = cfg.RPN.POSITIVE_FRACTION
        self.boundary_threshold = cfg.RPN.BOUNDARY_THRESH
        self.smooth_l1_beta = cfg.RPN.SMOOTH_L1_BETA

    def get_rpn_target(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        gt_objectness_logits = []
        gt_anchor_deltas = []
    
        for gt_boxes_i, anchors_i in zip(self.gt_boxes, self.anchors):
            match_quality_matrix = pairwise_iou(gt_boxes_i, anchors_i) #[num_gt_boxes, num_anchors]

            # Acnhors are assosiated with their ground truths, matched_idxs: [num_of_anchor], 
            # each element is index of gt with which the anchor is matched
            # gt_objectness_logits_i: [num_of_achors], each anchor is labeled as 0,-1,1. 
            # Note: Anchor with didn;t overlapped with any gt have been assigned garbage gt 
            # i.e their assigned gt index is of no significance. We will filter them later using 0,-1,1 
            matched_idxs, gt_objectness_logits_i = self.anchor_matcher(match_quality_matrix) 

            if (self.boundary_threshold >= 0) & (self.image_size is not None):
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors_i.inside_box(self.image_size, self.boundary_threshold)
                gt_objectness_logits_i[~anchors_inside_image] = -1

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                gt_anchor_deltas_i = torch.zeros_like(anchors_i.tensor)

            else:
                # TODO wasted computation for ignored boxes
                # matched_gt_boxes: Boxes of length num_anchors. Each element is corresponds to its matched gt_box
                matched_gt_boxes = gt_boxes_i[matched_idxs]

                # gt_anchor_deltas_i: Tensor(num_anchors, 4) - Have delta corresponding to each anchor. 
                gt_anchor_deltas_i = self.box2box_transform.get_deltas(
                    anchors_i.tensor, matched_gt_boxes.tensor
                )

            gt_objectness_logits.append(gt_objectness_logits_i)
            gt_anchor_deltas.append(gt_anchor_deltas_i)

        # This generates the labels and bbox regression target.
        # gt_objectness_logits: [batch_size, num_anchors]  gt_anchor_deltas: [batch_size, num_anchors, 4]
        return gt_objectness_logits, gt_anchor_deltas 
            

    def losses(self):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        # print(self.pred_objectness_logits.shape, self.pred_anchor_deltas.shape)
        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwriting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            pos_idx, neg_idx = subsample_labels(
                label, self.batch_size_per_image, self.positive_fraction, 0
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            return label

        # gt_objectness_logits: [batch_size, num_anchors]  gt_anchor_deltas: list of len batch_size and each element:[num_anchors, 4]   
        gt_objectness_logits, gt_anchor_deltas = self.get_rpn_target() 

        # Stack to: (N, num_anchors_per_image) 
        # In this gt_objectness_logits only 256 anchors in total have 1 and 0, others have been marked -1.
        # So those 256 are only gonna be used in training. 
        gt_objectness_logits = torch.stack(
            [resample(label) for label in gt_objectness_logits], dim=0
        )
       
        gt_objectness_logits = gt_objectness_logits.flatten() # shape: [batch_size*num_anchors_per_featmap=H*W*9] -> 1D tensor

        # Stack to: (N, num_anchors_per_image, B)
        gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0) # shape: [batch_size, num_anchors, 4]
        gt_anchor_deltas = gt_anchor_deltas.reshape(-1, 4) # shape: [tot_anchors_batch, 4]

        # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
        pred_objectness_logits = self.pred_objectness_logits.permute(0, 2, 3, 1).flatten()

        # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
        #          -> (N*Hi*Wi*A, B)
        x = self.pred_anchor_deltas
        pred_anchor_deltas = x.view(x.shape[0], -1, 4, x.shape[-2], x.shape[-1]).permute(0, 3, 4, 1, 2).reshape(-1, 4)
                
        objectness_loss, localization_loss = rpn_losses(
            gt_objectness_logits,
            gt_anchor_deltas,
            pred_objectness_logits,
            pred_anchor_deltas,
            self.smooth_l1_beta,
        )

        normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
        loss_cls = objectness_loss * normalizer  # cls: classification loss
        loss_loc = localization_loss * normalizer  # loc: localization loss
        losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}

        return losses

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        # print(self.pred_anchor_deltas.shape)

        B = self.anchors[0].tensor.size(-1)
        N, _ , Hi, Wi = self.pred_anchor_deltas.shape

         # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)    
        pred_anchor_deltas = self.pred_anchor_deltas.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
        
        anchors = Boxes.cat(self.anchors)
        proposals = self.box2box_transform.apply_deltas(pred_anchor_deltas, anchors.tensor)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
        proposals = proposals.view(N, -1, B)
        
        return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
        # print(self.pred_objectness_logits.shape)

        pred_objectness_logits = self.pred_objectness_logits.permute(0, 2, 3, 1).reshape(self.num_images, -1)

        return pred_objectness_logits

