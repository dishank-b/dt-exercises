'''
Backbone class for Faster-RCNN.
'''

import torch
import torchvision
import torch.nn as nn

from .resnet import *

class Backbone(nn.Module):
	"""docstring for Backbone"""
	def __init__(self, cfg):
		super(Backbone, self).__init__()
		
		self.model_name = cfg.BACKBONE.MODEL_NAME
		self.freeze_at = cfg.BACKBONE.FREEZE_AT

		if "resnet" in self.model_name:
			self.stop_layer = cfg.BACKBONE.RESNET_STOP_LAYER ## used for resnets only

		models = {'resnet50': resnet50, 'resnet18': resnet18}

		self.model = models[self.model_name](self.stop_layer, pretrained=True)
		
		#freezing the layers of network
		if self.freeze_at>0:
			self.freeze()

		if self.stop_layer==3:
			self.stride = 16
		elif self.stop_layer==4:
			self.stride = 32

		key =  'layer' + str(self.stop_layer) + '.1.conv1.weight'
		self.out_channels = self.model.state_dict()[key].shape[1] ## Number of output channels, to be used for RPN

	## Forward pass 
	def forward(self, image):
		return self.model(image)

	def freeze(self):
		if self.freeze_at==0:
			freeze_list = []
		elif self.freeze_at==1:
			freeze_list = ['conv1', 'bn1']
		elif self.freeze_at==2:
			freeze_list = ['conv1', 'bn1', 'layer1']
		
		for name, module in self.model.named_children():
			if name in freeze_list:
				# freeze weights and biases of mentioned layers
				for p_name, p in module.named_parameters():
					p.requires_grad = False
				# Freezing running mean and Varaince. 
				module.eval()

	def freeze_bn(self):
		for module in self.model.modules():
			if isinstance(module, nn.BatchNorm2d):
				module.weight.requires_grad = False
				module.bias.requires_grad = False
				module.eval()

