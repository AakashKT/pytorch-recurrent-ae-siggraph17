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
from data import *
from losses import *


def save_checkpoint(state, filename):
	torch.save(state, filename);

def train_sequence(model, sequence):
	output_final = sequence['B'].clone()
	output_final.fill_(0)
	target_final = sequence['B'].clone()
	target_final.fill_(0)

	inp = sequence['A']
	target = sequence['B']

	loss_final = 0
	ls_final = 0
	lg_final = 0
	lt_final = 0

	for j in range(0, 7):
		inpi = inp[:, j, :, :, :]
		gti = target[:, j, :, :, :]

		final_inp = {
			'A': inpi,
			'B': gti
		}

		model.set_input(final_inp)
		if j == 0:
			model.reset_hidden()

		output = model()
		output_final[:, j, :, :, :] = output
		target_final[:, j, :, :, :] = gti

	temporal_output, temporal_target = get_temporal_data(output_final, target_final)

	for j in range(0, 7):
		output = output_final[:, j, :, :, :]
		target = target_final[:, j, :, :, :]
		t_output = temporal_output[:, j, :, :, :]
		t_target = temporal_target[:, j, :, :, :]

		l, ls, lg, lt = loss_func(output, t_output, target, t_target)
		loss_final += l
		ls_final += ls
		lg_final += lg
		lt_final += lt

	return loss_final, ls_final, lg_final, lt_final


def train(model, dataset, optimizer, epoch):

	total_loss = 0
	total_loss_num = 0

	for i, item in enumerate(dataset):
		optimizer.zero_grad()
		loss_final, ls_final, lg_final, lt_final = train_sequence(model, item)
		
		loss_final.backward(retain_graph=False)
		optimizer.step()

		total_loss += loss_final.item()
		total_loss_num += 1

		if i % 50 == 0:
			print('[Epoch : %s] [%s/%s] Loss => %s , L1 => %s , HFEN => %s , TEMPORAL => %s' % \
					(epoch+1, (i+1), len(data_loader), loss_final.item(), ls_final.item(), \
						lg_final.item(), lt_final.item()))
			sys.stdout.flush()

	total_loss /= total_loss_num

	return total_loss



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='RecurentAE, SIGGRAPH \'17')
	parser.add_argument('--data_dir', type=str, help='Data directory')
	parser.add_argument('--save_dir', type=str, help='Model chekpoint saving directory')
	parser.add_argument('--name', type=str, help='Experiment Name')
	parser.add_argument('--epochs', type=int, help='Number of epochs to train')

	args = parser.parse_args()

	data_loader = RAEData('%s/train' % args.data_dir, (256, 256))
	dataset = DataLoader(data_loader, batch_size=1, num_workers=0, shuffle=True)

	model = RecurrentAE(8)
	model.to('cuda:0')
	print(model)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))

	for epoch in range(args.epochs):
		print('\nEpoch %s' % (epoch+1))

		total_loss = train(model, dataset, optimizer, epoch)

		print('Epoch %s loss => %s' % (epoch+1, total_loss))
		sys.stdout.flush()

		if epoch % 100 == 0:
			print('SAVING MODEL AT EPOCH %s' % (epoch+1))
			save_checkpoint({
					'epoch': epoch+1,
					'state_dict':model.state_dict(),
					'optimizer':optimizer.state_dict(),
				}, '%s/%s_%s.pt' % (args.save_dir, args.name, epoch+1))


	save_checkpoint({
				'epoch': args.epochs,
				'state_dict':model.state_dict(),
				'optimizer':optimizer.state_dict(),
			}, '%s/%s_%s.pt' % (args.save_dir, args.name, args.epochs))
