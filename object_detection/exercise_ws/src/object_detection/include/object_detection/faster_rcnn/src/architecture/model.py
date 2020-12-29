"""
Faster R-CNN class
"""

import sys
import torch
import torch.nn as nn

from src.backbone import Backbone
from src.rpn import RPN
from src.detection import Detector
from src.tracker import MultiObjTracker


class FasterRCNN(nn.Module):
	"""docstring for generalized_faster_rcnn"""
	def __init__(self, cfg):
		super(FasterRCNN, self).__init__()
		self.backbone = Backbone(cfg)
		# print(self.backbone)
		self.rpn = RPN(cfg, self.backbone.out_channels)
		self.detector = Detector(cfg, self.backbone.stride, self.backbone.out_channels)

	def freeze_not_sigma(self):
		print("freezing models")
		for name, param in self.named_parameters():
			if "sigma" not in name:
				param.requires_grad = False

	def forward(self, image, gt_target=None, is_training=False):
		"""
		Args:
			image: 	   Tensor[N,H,W,C], N is batch_size
			gt_target: Tensor[Instances], each instance has attribute 'gt_boxes' and 'gt_classes'
		
		Returns:
			rpn_proposals:  List[Instances], length of list is N. Each instance of the list have attribute 'proposal_boxes' and 'objectness_logits'
			rpn_losses: 	Dict, the dict have two losses - classification_loss and regression loss  
			prediction:		List[Instances], length of list is N. Each instance stores the topk most confidence detections. 
							It has attributes - 'pred_boxes', 'scores' , 'pred_sigma', 'pred_classes'
			detection_loss: Dict, classfication loss and bounding box loss

		"""

		image_size = image.shape[-2:]
		
		feature_map = self.backbone(image) # feature_map : [N, self.backbone_net.out_channels, H, W]

		rpn_proposals, rpn_losses = self.rpn(feature_map, gt_target, image_size, is_training) # topK proposals sorted in decreasing order of objectness score and losses: []
		
		detections, detection_loss = self.detector(feature_map, rpn_proposals, gt_target, is_training)

		return rpn_proposals, detections, rpn_losses, detection_loss 

class FasterRCNN_KF(nn.Module):
	"""docstring for generalized_faster_rcnn"""
	def __init__(self, cfg):
		super(FasterRCNN_KF, self).__init__()
		self.backbone = Backbone(cfg)
		self.rpn = RPN(cfg, self.backbone.out_channels)
		self.detector = Detector(cfg, self.backbone.stride, self.backbone.out_channels)
		self.tracker = MultiObjTracker(cfg)

	def forward(self, image, gt_target=None, is_training=False):
		"""
		Args:
			image: 	   Tensor[N,H,W,C], N is batch_size
			gt_target: Tensor[Instances], each instance has attribute 'gt_boxes' and 'gt_classes'
		
		Returns:
			rpn_proposals:  List[Instances], length of list is N. Each instance of the list have attribute 'proposal_boxes' and 'objectness_logits'
			rpn_losses: 	Dict, the dict have two losses - classification_loss and regression loss  
			prediction:		List[Instances], length of list is N. Each instance stores the topk most confidence detections. 
							It has attributes - 'pred_boxes', 'scores' , 'pred_sigma', 'pred_classes'
			detection_loss: Dict, classfication loss and bounding box loss

		"""

		image_size = image.shape[-2:]
		
		feature_map = self.backbone(image) # feature_map : [N, self.backbone_net.out_channels, H, W]

		rpn_proposals, rpn_losses = self.rpn(feature_map, gt_target, image_size, is_training) # topK proposals sorted in decreasing order of objectness score and losses: []
		
		detections, detection_loss = self.detector(feature_map, rpn_proposals, gt_target, is_training)
		# print("detections:" , [len(x) for x in detections])

		# _ = [x.draw("./logs/backpropkf/results", "target") for x in gt_target]

		for x,y,z in zip(detections, gt_target, rpn_proposals):
			x._image_path = y.image_path
			z._image_path = y.image_path

		# _ = [x[:50].draw("./logs/backpropkf/results", "proposals") for x in rpn_proposals]
		_ = [x.draw("./logs/testing/results", "detect") for x in detections]

		tracks, track_loss = self.tracker(detections, gt_target, is_training)

		return rpn_proposals, detections, tracks, rpn_losses, detection_loss, track_loss 

