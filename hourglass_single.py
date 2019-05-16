from __future__ import print_function, absolute_import
import sys

sys.path.insert(0, 'home/shantam/alexDarknet/darknet/hourglass/')
sys.path.insert(0, 'home/shantam/alexDarknet/darknet/hourglass/pose')
	
import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import _init_paths
#from pose import Bar
from hourglass.pose.utils.logger import Logger, savefig
from hourglass.pose.utils.evaluation import accuracy, AverageMeter, final_preds

from hourglass.pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from hourglass.pose.utils.osutils import mkdir_p, isfile, isdir, join
from hourglass.pose.utils.imutils import batch_with_heatmap, im_to_numpy
from hourglass.pose.utils.transforms import fliplr, flip_back
import hourglass.pose.models as models
import hourglass.pose.datasets as datasets
import hourglass.pose.losses as losses

from hourglass.pose.utils.transforms import *

import cv2, nonechucks as nc, numpy as np 

def highest(scores, values):
		maxval = np.amax(values)
		maxindex = np.argmax(values)
		values = np.delete(values, maxindex)
		return values, maxindex

class Hourglass():

	def __init__(self):

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		cudnn.benchmark = True

		self.img_path=''
		self.dataset = 'mpii'
		self.image_path = ''
		self.inp_res = 256
		self.out_res = 64

		self.arch = 'hg'
		self.stacks = 2
		self.blocks = 1
		self.features = 256
		self.resnet_layers = 50

		self.solver = 'rms'
		self.workers = 1
		self.epochs = 100
		self.test_batch=1
		self.train_batch=1
		self.lr = 2.5e-4
		self.momentum=0
		self.weight_decay=0
		self.gamma=0.1

		self.sigma=1.0
		self.scale_factor=0.25
		self.rot_factor=1
		self.sigma_decay=0

		#self.checkpoint=''
		self.resume=''
		self.njoints=24

		self.model = models.hg(num_stacks=self.stacks, num_blocks=self.blocks, num_classes=self.njoints, resnet_layers=self.resnet_layers)

		self.model = torch.nn.DataParallel(self.model).to(self.device)
		self.criterion = losses.JointsMSELoss().to(self.device)
		self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

		self.checkpoint = torch.load("/home/shantam/Documents/Programs/hourglass/checkpoint/mpii/hg_updated_21/checkpoint.pth.tar")
		self.start_epoch = self.checkpoint['epoch']
		self.model.load_state_dict(self.checkpoint['state_dict'])
		self.model.eval()

	def forward_pass(self, img):

		#img=img.unsqueeze(0)
		#img=[img]
		#img=np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		points = []
		pointers = []

		#image = im_to_numpy(img)

		c = [img.shape[1]/2, img.shape[2]/2]
		s = float(img.shape[1]/200.0)
	
		#print ("cropped", c, s)
		img = crop(self.img_path, img, c, s, [self.inp_res, self.inp_res])
		
		#image = im_to_numpy(img)

		# while True:
		# 	#print (type(image))
		# 	image=img.cpu().numpy()
		# 	image=np.moveaxis(image, 0, -1)
		# 	image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		# 	cv2.imshow('scaled', image)
		# 	cv2.waitKey(10)
			
		
		img=np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
		img = img.to(device, non_blocking = True)
		output=self.model(img)
	
		score_map = output[-1].cpu() if type(output) == list else output.cpu()
		preds, vals = final_preds(score_map, [c], [s], [64, 64])
		coords = np.squeeze(preds)

		for m in range(0,len(coords)):
			val = vals[0][m].detach().numpy()
			print ("val",val)
			if val>0.4: #threshold for confidence score
				x,y = coords[m][0].cpu().detach().numpy(), coords[m][1].cpu().detach().numpy()
				if ([x,y] != present for present in pointers):
					#print ("coming in here")
					pointers.append([x,y,m])

				else:
					for present in pointers:
						if [present[0], present[1]]==[x,y]:
							if val>present[2]:
								pointers.remove(present)
								pointers.append([x,y,m])
	
		finalpoints=[]
		finalpointers=[]
		for j in pointers:
			x,y,m = j[0], j[1], j[2]
			finalpointers.append([j[0], j[1]])
			finalpoints.append(j[2])

		return finalpoints, finalpointers

