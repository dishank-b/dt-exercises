import torch
from PIL import Image
import numpy as np
import json
from torch.utils.data import Dataset 
import os
import glob
from ..utils import Boxes, Instances
import time
from tqdm import tqdm

class DuckieDataset(Dataset):
	"""Kitti Dataset Reader."""

	def __init__(self, root_dir,transform=None, cfg = None):
		"""
		Args:
			root_dir (string): Path to the dataset.

			transform (callable, optional): Optional transform to be applied
				on a sample.

			cfg: config file
		"""

		self.cfg = cfg
		self.root_dir = root_dir
		self.transform = transform

		self.data_dict = self._makedata(self.root_dir)
		self.data_keys = list(self.data_dict.keys())

	def _makedata(self, root_dir):
		data= {}
		files_names = glob.glob(os.path.join(root_dir,"*.npz"))
		files_names = sorted(files_names)

		for name in tqdm(files_names):
			data_img = np.load(name)
			img = data_img[f"arr_{0}"]
			boxes = data_img[f"arr_{1}"]
			classes = data_img[f"arr_{2}"]
			# print(name, img.shape, boxes.shape, classes.shape)
			if len(boxes)!=0:
				target = Instances(img.shape[:2], gt_boxes=Boxes(torch.tensor(boxes)), gt_classes=torch.tensor(classes))
				data[name] = (img, target)

		return data	

	def __len__(self):
		return len(self.data_dict)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample= {}
		img_path = self.data_keys[idx]
		sample["image_path"] = img_path
		sample["target"] = self.data_dict[img_path][1]
		img = self.data_dict[img_path][0]
		
		# transform the image
		if self.transform:
			img = self.transform(img)

		sample["image"] = img

		return sample

	# A collate function to enable loading in batch
	def collate_fn(self, batch): #batch is the list of samples

		elem = batch[0]
		batch = {key:[x[key] for x in batch] for key in elem}
		batch["image"] = torch.stack(batch["image"], dim=0)

		return batch

