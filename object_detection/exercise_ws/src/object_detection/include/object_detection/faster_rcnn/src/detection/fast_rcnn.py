# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..nms import batched_nms, bayes_od_clustering
from ..utils import Boxes, Matcher, Box2BoxTransform, Instances
from ..loss import smooth_l1_loss

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_bbox_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_bbox_deltas: ground-truth box2box transform deltas
"""

class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, cfg, input_shape, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        super(FastRCNNOutputLayers, self).__init__()

        fc_dim     = cfg.ROI_HEADS.FC_DIM

        self.fc1 = nn.Linear(np.prod(input_shape), fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)

        self.cls_score = nn.Linear(fc_dim, num_classes + 1)

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(fc_dim, num_bbox_reg_classes * box_dim)
        
        for l in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def freeze(self):
        for name, module in self.named_children():
            # freeze weights and biases of mentioned layers
            for p_name, p in module.named_parameters():
                if "sigma" not in p_name:
                    p.requires_grad = False

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        for layer in [self.fc1, self.fc2]:
            x = F.relu(layer(x))

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return tuple(list(x) for x in zip(*result_per_image))


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    if num_bbox_reg_classes == 1:
        # We have two approaches do this. 
        # Approach 1 (commented out)- 
        # Basically one box is counted multiple times, each for 
        # class which have score > score_thresh.
        
        # # Filter results based on detection scores
        # filter_mask = scores > score_thresh  # R x K    
        # # R' x 2. First column contains indices of the R predictions;
        # # Second column contains indices of classes.
        # # all the 2D indices for which filter_mask is true.
        # filter_inds = filter_mask.nonzero()
        # # R' x 4 - Wehave repeated box corresponding to 
        # # all classes which have > score_thresh for each box
        # boxes = boxes[filter_inds[:, 0], 0]
        # variance = variance[filter_inds[:,0 ], 0]
        # scores = scores[filter_mask]
        # class_inds = filter_inds[:,1]
        # filter_inds = filter_inds[:,0]
        
        # Approach 2 -
        # In this approach we correspond the box only to the class
        # with max score. 
        # chossing the class with max score, class_ind for each box
        # scores: Rx1 - max score for each box
        # class_inds: Rx1 - index of class with max score
        scores, class_inds = torch.max(scores, 1)
        #Filter the boxes with the score_thresh
        filter_mask = scores > score_thresh
        filter_inds = filter_mask.nonzero().squeeze(1)
        # print(filter_inds)
        assert filter_inds.dim()==1, filter_inds.shape
        
        scores = scores[filter_inds]
        class_inds = class_inds[filter_inds]
        boxes = boxes[filter_inds,0]

    else:
        # Filter results based on detection scores
        filter_mask = scores > score_thresh  # R x K    
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        # all the 2D indices for which filter_mask is true.
        filter_inds = filter_mask.nonzero()
        
        boxes = boxes[filter_mask]
        scores = scores[filter_mask]
        class_inds = filter_inds[:,1]
        filter_inds = filter_inds[:,0]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, class_inds, nms_thresh)

    # if topk_per_image >= 0:
    #     keep = keep[:topk_per_image]

    use_bayesian_clustering = False
    # use_bayesian_clustering = True

    if use_bayesian_clustering==True & len(boxes)!=0:
        boxes, scores, class_inds = bayes_od_clustering(keep, boxes, scores, class_inds, nms_thresh)
    else:
        boxes,  scores, class_inds, filter_inds = boxes[keep], scores[keep], class_inds[keep], filter_inds[keep]
    
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = class_inds

    return result, filter_inds


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self, box2box_transform, pred_class_logits, pred_bbox_deltas, proposals, smooth_l1_beta, loss_type
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_bbox_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_bbox_deltas = pred_bbox_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.loss_type = loss_type

        box_type = type(proposals[0].proposal_boxes)

        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            # concatenates over all images in the batch
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        # self._log_accuracy()
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")


    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        gt_bbox_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        ) 
        box_dim = gt_bbox_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_bbox_deltas.size(1) == box_dim
        device = self.pred_bbox_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_bbox_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_bbox_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_bbox_deltas[fg_inds[:, None], gt_class_cols],
            gt_bbox_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg


    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """

        regress_loss_dir = {"deterministic": self.smooth_l1_loss}
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": regress_loss_dir[self.loss_type](),
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_bbox_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_bbox_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """

        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )

