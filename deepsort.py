import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.05
session = tf.Session(config=config)

from deep_sort import generate_detections
from vehicle_vgg import vehicle_encoder

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker 
from deep_sort.application_util import preprocessing as prep
from deep_sort.application_util import visualization
from deep_sort.deep_sort.detection import Detection

import keras
import numpy as np

import matplotlib.pyplot as plt

class deepsort_rbc():
	def __init__(self):
		#loading this encoder is slow, should be done only once.
		#self.encoder = generate_detections.create_box_encoder("deep_sort/resources/networks/mars-small128.ckpt-68577")		
		self.encoder = vehicle_encoder('weights/vehicle_vgg.h5')
		self.metric = nn_matching.NearestNeighborDistanceMetric("cosine",.5 , 100)
		self.tracker= Tracker(self.metric)

	def reset_tracker(self):
		self.tracker= Tracker(self.metric)

	#Older yolo used to give y,x,h,w. 	
	"""
	def format_yolo_output( self,out_boxes):
		out_boxes=np.array([out_boxes[:,1],out_boxes[:,0],\
			out_boxes[:,3]-out_boxes[:,1],out_boxes[:,2]-out_boxes[:,0]])
		out_boxes=out_boxes.T
		return out_boxes				
	"""

	#Current Yolo gives x_center,y_center,w,h
	#Deep sort needs the format `top_left_x, top_left_y, width,height
	
	def format_yolo_output( self,out_boxes):
		for b in range(len(out_boxes)):
			out_boxes[b][0] = out_boxes[b][0] - out_boxes[b][2]/2
			out_boxes[b][1] = out_boxes[b][1] - out_boxes[b][3]/2
		return out_boxes				


	def run_deep_sort(self, frame, out_scores, out_boxes, out_classes):
		out_boxes = self.format_yolo_output(out_boxes)
		
		if out_boxes==[]:			
			self.tracker.predict()
			trackers = self.tracker.tracks
			return trackers

		detections = np.array(out_boxes)
		#features = self.encoder(frame, detections.copy())
		features = self.encoder.extract_features(frame,detections)

		#print(frame.shape)

		detections = [Detection(bbox, score, feature,classname) \
					for bbox,score, feature,classname in\
				zip(detections,out_scores, features,out_classes)]

		outboxes = np.array([d.tlwh for d in detections])

		outscores = np.array([d.confidence for d in detections])
		indices = prep.non_max_suppression(outboxes, 0.8,outscores)
		
		detections = [detections[i] for i in indices]
		self.tracker.predict()
		self.tracker.update(detections)	
		trackers = self.tracker.tracks

		return trackers



