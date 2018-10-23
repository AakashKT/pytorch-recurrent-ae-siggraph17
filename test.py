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
import torch, argparse, pdb

from model import *


def load_checkpoint(filename):
	chkpoint = torch.load(filename);
	model = RCNN(8);
	model.to('cuda:0');
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

	epoch = chkpoint['epoch'];
	model.load_state_dict(chkpoint['state_dict']);
	optimizer.load_state_dict(chkpoint['optimizer']);

	return model, optimizer, int(epoch);


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='RecurentAE, SIGGRAPH \'17')
	parser.add_argument('--data_dir', type=str, help='Data directory')
	parser.add_argument('--output_dir', type=str, help='Directory to save output')
	parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')

	args = parser.parse_args()

	model, optimizer, epoch = load_checkpoint(args.checkpoint)

	size = (256, 256)
	width = size[0]
	height = size[1]

	data_loader = RCNNData('%s/test' % args.data_dir, size)
	dataset = DataLoader(data_loader, batch_size=1, num_workers=0, shuffle=False)

	for i, item in enumerate(dataset):

		os.system('mkdir -p %s/seq_%s' % (args.output_dir, i))
		for j in range(0, 7):
			inp = item['A']
			gt = item['B']

			inp = inp[:, j, :, :, :]
			gt = gt[:, j, :, :, :]

			final_inp = {
				'A': inp,
				'B': gt
			}

			model.set_input(final_inp)
			if j == 0:
				model.reset_hidden()

			output = model()
		
			albedo = item['ALBEDO'].clone()
			albedo = albedo[:, j, :, :, :]
			albedo = torch.squeeze(albedo.detach(), dim=0) * 255.0
			albedo = albedo.permute((1, 2, 0))
			albedo = albedo.cpu().numpy()

			ray = final_inp['B'].clone()
			ray = torch.squeeze(ray, dim=0)
			ray = ray[:3, :, :]
			ray = ray.permute((1, 2, 0))
			ray = ray.cpu().numpy()
			ray *= 255.0

			output = torch.squeeze(output.detach(), dim=0)
			output = output.permute((1, 2, 0))
			output = output.cpu().numpy()
			output *= 255.0

			og = final_inp['A']
			og = torch.squeeze(og.detach(), dim=0) * 255.0
			og = og.permute((1, 2, 0))
			og = og.cpu().numpy()

			final = np.zeros((height, width * 4, 3), dtype=np.float)
			final[:, :width, :] = og[:, :, :3]
			final[:, width:width * 2, :] = albedo
			final[:, width * 2:width * 3, :] = output
			final[:, width * 3:width * 4, :] = ray

			cv2.imwrite('%s/seq_%s/%s.png' % (args.output_dir, i, j), final)
