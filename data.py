import torch, os, sys, cv2
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import torchvision.transforms as transforms
import numpy as np 
import torch


class RAEData(Dataset):

	def __init__(self, input_dir, size):
		super(RAEData, self).__init__()
		
		self.input_dir = input_dir
		self.images = sorted(os.listdir(self.input_dir))

		self.width = size[0]
		self.height = size[1]

	def __getitem__(self, index):
		# 10spp_shading   ray_shading
		# 10spp_albedo    normal 
		# depth           roughness
		#-----------------------------

		A = np.zeros((7, self.height, self.width, 8), dtype=np.float)
		B = np.zeros((7, self.height, self.width, 3), dtype=np.float)
		ALBEDO = np.zeros((7, self.height, self.width, 3), dtype=np.float)

		seq_images = sorted(os.listdir('%s/%s' % (self.input_dir, self.images[index])))
		for i, item in enumerate(seq_images):
			img = cv2.imread('%s/%s/%s' % (self.input_dir, self.images[index], item))
			img = cv2.resize(img, (self.width * 2, self.height * 3))

			shading = img[:self.height, :self.width, :]
			ray_shading = img[:self.height, self.width:, :]
			albedo = img[self.height:self.height * 2, :self.width, :]
			normal = img[self.height:self.height * 2, self.width:, :]
			depth = (img[self.height * 2:, :self.width, 0] + img[self.height * 2:, :self.width, 1] \
								+ img[self.height * 2:, :self.width, 2]) / 3
			roughness = (img[self.height * 2:, self.width:, 0] + img[self.height * 2:, self.width:, 1] \
								+ img[self.height * 2:, self.width:, 2]) / 3
			depth = np.expand_dims(depth, axis=2)
			roughness = np.expand_dims(roughness, axis=2)

			ray_shading = ray_shading.astype(np.float) / 255.0
			shading = shading.astype(np.float) / 255.0
			normal = normal.astype(np.float) / 255.0
			albedo = albedo.astype(np.float) / 255.0
			depth = depth.astype(np.float) / 255.0
			roughness = roughness.astype(np.float) / 255.0

			A[i, :, :, :3] = shading
			A[i, :, :, 3:6] = normal
			A[i, :, :, 6:7] = depth
			A[i, :, :, 7:8] = roughness

			B[i, :, :, :] = ray_shading
			ALBEDO[i, :, :, :] = albedo

		A = torch.from_numpy(A)
		B = torch.from_numpy(B)
		ALBEDO = torch.from_numpy(ALBEDO)

		A = A.permute((0, 3, 1, 2))
		B = B.permute((0, 3, 1, 2))
		ALBEDO = ALBEDO.permute((0, 3, 1, 2))

		return {
			'A': A.type(torch.float).to('cuda:0'),
			'B': B.type(torch.float).to('cuda:0'),
			'ALBEDO': ALBEDO.type(torch.float).to('cuda:0')
		}

		
	def __len__(self):
		return len(self.images)

	def np_normalize(self, img):
		return (img - img.min()) / (img.max() - img.min())

	def save_image(self, img, img_name):
		img = torch.squeeze(img.detach(), dim=0) * 255.0
		img = img.permute((1, 2, 0))
		img = img.cpu().numpy().astype(np.uint8)
		
		cv2.imwrite(img_name, img)