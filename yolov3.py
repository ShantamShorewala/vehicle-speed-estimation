from ctypes import *
import math
import random
import cv2
import os
import numpy as np
from random import randint

from pydarknet import *

class YOLOV3(object):
		
	@classmethod		
	def load_model(self,cfg_path,weights_path,data_path):
		#cfg_path=bytes(cfg_path, encoding='utf-8')
		#weights_path=bytes(weights_path, encoding='utf-8')
		#data_path=bytes(data_path, encoding='utf-8')
		self.net = load_net(cfg_path,weights_path, 0)
		self.meta = load_meta(data_path)

	"""docstring for YOLOV3"""
	def __init__(self,cfg_path,weights_path,data_path):
		super(YOLOV3, self).__init__()
		self.load_model(cfg_path,weights_path,data_path)



	def sample(self,probs):
		s = sum(probs)
		probs = [a/s for a in probs]
		r = random.uniform(0, 1)
		for i in range(len(probs)):
			r = r - probs[i]
			if r <= 0:
				return i
		return len(probs)-1

	def c_array(self,ctype, values):
		arr = (ctype*len(values))()
		arr[:] = values
		return arr


	def classify(self,im):
		out = predict_image(self.net,im)
		res = []
		for i in range(self.meta.classes):
			res.append((self.meta.names[i], out[i]))
		res = sorted(res, key=lambda x: -x[1])
		return res


	def array_to_image(self,arr):
		# need to return old values to avoid python freeing memory
		arr = arr.transpose(2,0,1)
		c, h, w = arr.shape[0:3]
		arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
		data = arr.ctypes.data_as(POINTER(c_float))
		im = IMAGE(w,h,c,data)
		return im, arr



	def detect_image(self,image, thresh=.8, hier_thresh=.5, nms=.45):

		im, image = self.array_to_image(image)
		rgbgr_image(im)
		num = c_int(0)
		pnum = pointer(num)
		predict_image(self.net, im)
		dets = get_network_boxes(self.net, im.w, im.h, thresh, 
								 hier_thresh, None, 0, pnum)
		num = pnum[0]
		if nms: do_nms_obj(dets, num, self.meta.classes, nms)

		res = []
		out_scores = []
		out_boxes = []
		out_classes = []


		for j in range(num):
			a = dets[j].prob[0:self.meta.classes]
			if any(a):
				ai = np.array(a).nonzero()[0]
				for i in ai:
					b = dets[j].bbox
					out_scores.append(dets[j].prob[i])
					out_classes.append(self.meta.names[i].strip('\r'))
					out_boxes.append([b.x, b.y, b.w, b.h])

					#res.append((self.meta.names[i].strip('\r'), dets[j].prob[i],(b.x, b.y, b.w, b.h)))

		#res = sorted(res, key=lambda x: -x[1])

		if isinstance(image, bytes): free_image(im)
		free_detections(dets, num)

		#return res
		return out_scores, out_boxes, out_classes
