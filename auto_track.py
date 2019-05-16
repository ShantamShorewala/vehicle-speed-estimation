import cv2
from scipy import misc
import numpy as np

from yolov3 import YOLOV3
from deepsort import deepsort_rbc

from pydarknet import *
from collections import defaultdict
import time

class analytics_rbc:

	def __init__(self, line, sline1, sline2):
		#line is for counting
		#slines for speed estimation
		self.deepsort=deepsort_rbc()
		self.yolov3 = YOLOV3("/home/shantam/alexDarknet/darknet/weights_2903/yolov3_rbc.cfg","/home/shantam/alexDarknet/darknet/weights_2903/weights2/yolov3_rbc_180000.weights","/home/shantam/alexDarknet/darknet/weights_2903/ta.data")

		#New yolov3 model
		#self.yolov3 = YOLOV3("cfg/yolov3_rbc.cfg","weights/yolov3_rbc_60000.weights","cfg/2k_aug.data")		


		#self.below_line=set()
		#self.above_line=set()
		#self.considered=set()
		#self.speed1=set()
		#self.idtocoords=dict()
		
		self.classes = ['car','bus','two-wheeler','three-wheeler','people','lcv','bicycle','truck']
		#self.v_count_up= defaultdict(int)
		#self.v_count_down= defaultdict(int)

		#Initialize each value to zero
		#for c in self.classes:
		#	self.v_count_up[c]
		#	self.v_count_down[c]	

		#self.sline1=sline1
		#self.sline2=sline2
		#self.line=line
		#self.tot_dist=0
		#self.tot_time=0
		#self.speed=0
		#self.speed_dict = defaultdict(int)
		#self.frame_count = defaultdict(int)

	def calc_line_point(self, xy0, xy1, point_cord,frame):

			if((xy1[1]-xy0[1])==0 or (xy1[1]-xy0[1])==0):
				print("coordinates give a point and not a line!!")
			try:
				slope = (xy1[1]-xy0[1])/(xy1[0]-xy0[0])

			except(ZeroDivisionError):
					print("coordinates give a point and not a line!!")
					exit(0)
			
			"""Determinant method"""
			d = (point_cord[0]-xy0[0])*(xy1[1]-xy0[1]) - (point_cord[1] - xy0[1])*(xy1[0]-xy0[0])
			cv2.line(frame,(xy0[0],xy0[1]),(xy1[0],xy1[1]),(0,0,0),2)
			height,width = frame.shape[:2]
			y1old = xy0[1]
			y2old = xy1[1]
			const = xy0[1]-(slope*xy0[0])
			y1new = int(slope*0 + const)
			y2new = int(slope*width + const)
			#cv2.line(frame,(0,y1new),(width,y2new),(0,0,0),2)
			#d = (point_cord[0] - 0)*(y2new - y1new) - (point_cord[1] - y1new)*(width - 0)

			if (d==0):
					return 0
			elif (d<0):
					#above the line
					return 1
			elif (d>0):
					"""Below the line"""
					return -1


	def count_id(self, bbox, id_num, classname,frame):
			#End points of the line
			p1 = self.line[0]
			p2 = self.line[1]
			bbox_hat = [0, 0]

			#Centroid
			bbox_hat[0] = int((bbox[0] + bbox[2])/2)
			bbox_hat[1] = int((bbox[1] + bbox[3])/2)

			point_position = self.calc_line_point(p1, p2, (bbox_hat[0],bbox_hat[1]),frame)
			
			#Zero is impossible!
			if(point_position==1):
				if id_num in self.below_line:
					self.v_count_up[classname]+=1
					self.below_line.remove(id_num)
					self.considered.add(id_num)
				else:
					self.above_line.add(id_num)

			elif(point_position==-1):
				if(id_num in self.above_line):
					self.v_count_down[classname]+=1
					self.above_line.remove(id_num)
					self.considered.add(id_num)
				else:
					self.below_line.add(id_num)


	def calc_speed(self,bbox, id_num,cross_time,frame):
		# time=(time/1000)
		#print(bbox)
		#self.speed = 0

		box_center_x = (bbox[0] + bbox[2]) / 2
		box_center_y = (bbox[1] + bbox[3]) / 2

		point_position_sline1 = self.calc_line_point(self.sline1[0],self.sline1[1],(box_center_x,box_center_y),frame)		
		point_position_sline2 = self.calc_line_point(self.sline2[0],self.sline2[1],(box_center_x,box_center_y),frame)		


		if (point_position_sline1==-1) and (point_position_sline2==-1) and self.speed_dict[id_num]!=0:
			print("Here ",id_num)
			return self.speed_dict[id_num]

		if point_position_sline1 + point_position_sline2==0:
			#Calculate no of frames spent inside the 2 lines
			self.frame_count[id_num]+=1			

		if point_position_sline1 ==-1 and id_num not in self.speed1:
			self.speed1.add(id_num)
			#self.idtocoords[id_num]=(box_center_x,box_center_y,cross_time)
			self.idtocoords[id_num]=(box_center_x,box_center_y,time.time())
			self.frame_count[id_num]+=1			

		elif point_position_sline2 ==-1 and id_num in self.speed1:
			self.speed1.remove(id_num)
			x1,y1,t1=self.idtocoords[id_num]
			self.idtocoords.pop('key', None)
			#x2,y2,t2=box_center_x,box_center_y,cross_time
			x2,y2,t2=box_center_x,box_center_y,time.time()
			self.frame_count[id_num]+=1

			if t1==t2:
				print("same time")

			if(t1!=t2):
				#print(x1,y1,x2,y2,t2,t1)
				print(t2,t1,' time ')
				self.tot_dist = 9.5 #Meters
				self.tot_time = (t2-t1) #*5#FPS
				self.speed_dict[id_num] = float(self.tot_dist)/self.tot_time
				#FPS = 5
				#self.speed_dict[id_num] = (FPS*self.tot_dist)/self.frame_count[id_num]


		print('ID NUM', id_num,self.speed1)
		print(point_position_sline1,point_position_sline2,'Speed: ',self.speed_dict[id_num],'tot_time: ',self.tot_time)
		print
	
		"""
		if(bbox[3]<self.sline2 and bbox[1]<self.sline2 and id_num not in self.speed1):
			self.speed1.add(id_num)
			self.idtocoords[id_num]=((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, time)

		elif(bbox[3]<self.sline1 and bbox[1]<self.sline1 and id_num in self.speed1 ):
			self.speed1.remove(id_num)
			x1,y1,t1=self.idtocoords[id_num]
			self.idtocoords.pop('key', None)
			x2,y2,t2=(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2 , time
			if(t1!=t2):
				self.tot_dist+=(((x1-x2)**2+(y1-y2)**2)**0.5)
				self.tot_time+=t2-t1
				self.speed=float(self.tot_dist)/self.tot_time
		"""

		#return self.speed
		return self.speed_dict[id_num]


	def reset_instance(self):
		self.tot_dist=0
		self.tot_time=0
		self.v_count_up= defaultdict(int)
		self.v_count_down= defaultdict(int)

		#Initialize each value to zero
		for c in self.classes:
			self.v_count_up[c]
			self.v_count_down[c]	


	def run_analytics(self,frame,time_old):
		yolo_frame = frame
		out_scores, out_boxes, out_classes = self.yolov3.detect_image(yolo_frame)
		ret_struct=[]

		#print (out_classes, out_boxes, out_scores)
		out_classes2, out_boxes2, out_scores2 = [], [], []
		#out_classes = [x for x in out_classes if x == 'three-wheeler']
		for i in range(0, len(out_classes)):
			if out_classes[i] == 'three-wheeler':
				out_classes2.append(out_classes[i])
				out_boxes2.append(out_boxes[i])
				out_scores2.append(out_scores[i])
		##print (out_scores2)	
		#time.sleep(1.5)
	
		trackers = self.deepsort.run_deep_sort(frame,out_scores2,out_boxes2,out_classes2)
		#speed_for_vehicle = 0

		#print len(trackers)
		for track in trackers:
			#time.sleep(5)
			if not track.is_confirmed() or track.time_since_update > 1:
				continue
			
			temp_dict={}
			bbox=track.to_tlbr()

			#Important. Else this will screw up the dict.
			classname= track.classname
			classname = classname.decode('utf8').strip('\r')

			id_num = str(track.track_id)
			temp_dict['id']=id_num
			temp_dict['bbox']=bbox
			temp_dict['classname']=classname
			ret_struct.append(temp_dict)
			#self.count_id(bbox,id_num,classname,frame)
			#speed_for_vehicle = self.calc_speed(bbox,id_num,time.time(),frame)
			#temp_dict['speed'] = speed_for_vehicle

		#return_dict={'up_count':self.v_count_up, 'down_count':self.v_count_down, 'speed':speed_for_vehicle}
		#return return_dict, ret_struct
		return ret_struct