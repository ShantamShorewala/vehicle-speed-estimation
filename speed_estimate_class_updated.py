from __future__ import print_function, absolute_import
import cv2
#from deepsort import deepsort_rbc
from yolov3 import YOLOV3
import cv2
import os
import time, math
import numpy as np
from scipy.optimize import least_squares, leastsq 
#from auto_track import analytics_rbc
import hourglass_single #.example.hourglass_single #import hourglass
import csv



class speed_estimation:

	def __init__(self, img_height, img_width):

		self.time = 1.0/20.0 #set to time duration between two consecutive frames
		self.tracked = []
		self.last = []
		self.tracking = []

		self.framecount=0
		self.frame_number=0

		self.model = hourglass_single.Hourglass()
		self.prev_frame = np.zeros((img_width, img_height))


	def flip(self, pos):
		
		if pos[2]<0:
			pos*=-1
		return pos

	def distance(self, last_pos, current_pos):

		val=0
		for i in range(0,3):
			val += (last_pos[i]-current_pos[i])*(last_pos[i]-current_pos[i])
		val=math.pow(val,0.5)
		return val

	def compute_pose(self, points, imagepoints):
	
		pts={0: [45.0, 256.5, 71.0], 1: [-45.0, 256.5, 71.0], 2: [56.0, 186.7, 89.0], 3: [-56.0, 186.7, 89.0], 4: [51.0, 240.0, 149.5], 5: [-51.0, 240.0, 149.5], 6: [0.0, 260.0, 41.5], 7: [-47.5, 43.6, 104.7], 8: [-57.0, 57.8, 145.5], 9: [-54.3, 135.0, 149.8], 10: [-66.4, 220.0, 19.8], 11: [-43.0, 68.0, 165.6], 12: [57.0, 57.8, 145.5], 13: [54.3, 135.0, 149.8], 14: [43.0, 68.0, 165.6], 15: [47.5, 43.6, 104.7], 16: [66.4, 220.0, 19.8], 17: [44.2, 61.5, 152.4], 18: [-44.2, 61.5, 152.2], 19: [49.0, 56.0, 116.5], 20: [-49.0, 56.0, 116.5], 21: [0.0, 0.0, 38.0], 22: [33.0, 36.0, 83.5], 23: [-33.0, 36.0, 83.5]}
		'''order of pts: rear light (left), rear light (right), greenlow corner left, greenlow corner right, top corner rear left, top corner rear right, rear center, indicator light right, 
		right mirror, right center pole top, right wheel, top corner front right, left mirror, left center pole top, top corner front left , indicator light left, left wheel, Wind shield(top left)
		Wind shield(top right), Wind shield(bottom left), Wind shield(bottom right), front bonet, head light left, Head light right'''

		#insert new camera matrix here
		mtx = np.array([[1324.110551, 0.000000, 993.993108], [0.000000, 1324.110210, 621.997610],[  0. ,   0. ,   1. ]])
		dist = np.array([[-0.401747, 0.148985, -0.008159, -0.006626, 0.000000]]) 
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		worldpoints = np.zeros([len(points),3])
		twist = np.zeros([len(points),2])

		j=0
		for i in points:
			worldpoints[j] = pts[i]
			j+=1

		for i in range(len(imagepoints)):
			twist[i] = [float(imagepoints[i][0]), float(imagepoints[i][1])]

		_, rvecs, tvecs = cv2.solvePnP(worldpoints, twist, mtx, dist)
		tvecs = self.flip(tvecs)

	# mean_error = 0
	# for i in range(len(points)):
	# 	imgpoint = np.array([float(imagepoints[i][0]), float(imagepoints[i][1])])
	# 	ponting = np.array([[pts[i]]], dtype=np.float)
	# 	imgpoints2, _ = cv2.projectPoints(ponting, rvecs, tvecs, mtx, dist)
	# 	error = cv2.norm(imgpoint, imgpoints2[0][0], cv2.NORM_L2)/len(imgpoints2)
	# 	mean_error += error
	# old_error = mean_error/len(points)

	# placeholder = np.array([0, 0, 0, 0, 0, 0], dtype=np.float)
	# for i in range(len(rvecs)):
	# 	placeholder[i]=rvecs[i]
	# 	placeholder[i+3]=tvecs[i]

	# twist_reshaped = twist.reshape(-1)
	# print (twist_reshaped.shape)
	# pose_opt = leastsq(reproject_error, placeholder, args = (points, twist_reshaped), maxfev=10)
	# pose_opt = np.asarray(pose_opt)
	# tvecs_opt, rvecs_opt = pose_opt[0][3:6], pose_opt[0][0:3]
	# tvecs_opt = flip(tvecs_opt)

	# mean_error = 0
	# for i in range(len(points)):
	# 	imgpoint = np.array([float(imagepoints[i][0]), float(imagepoints[i][1])])
	# 	ponting = np.array([[pts[i]]], dtype=np.float)
	# 	imgpoints2, _ = cv2.projectPoints(ponting, rvecs_opt, tvecs_opt, mtx, dist)
	# 	error = cv2.norm(imgpoint, imgpoints2[0][0], cv2.NORM_L2)/len(imgpoints2)
	# 	mean_error += error
	# new_error = mean_error/len(points)

	# print ("translation", old_error - new_error, old_error, new_error)
	# #time.sleep(1)

	# if new_error<=old_error:
	# 	return tvecs_opt
	# else:
	# 	return tvecs

		return tvecs

	def draw_boxes(self, frame, class_boxes, speed, onlyspeed):
 
		for item in class_boxes:
			classname = item['classname']
			bbox = item['bbox']
			id_number = item['id']
			#speed = item['speed']

			bbox = list(map(int,bbox))

			x1,y1,x2,y2 = bbox
			#print ("well", x1, x2, y1, y2)
			#time.sleep(1)

			#cv2.line(frame,count_line[0],count_line[1],(0,0,0),2)
			cropped = frame[y1:y2, x1:x2]
			cv2.rectangle(frame, (x1,y1),(x2,y2),(255, 255, 0),3)
			cv2.putText(frame, "ID:"+str(id_number) , (x2,y2),cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0),2, cv2.LINE_AA)
			cv2.putText(frame, str(classname.decode("utf-8")), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX,1.15, (0, 255, 255),2)
			if (onlyspeed==True):
				cv2.putText(frame, "Speed:"+str(round(speed, 2))+ "m/s" , (x2+50,y2), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0),2, cv2.LINE_AA)
			
		return frame, cropped

	def init_trackers(self, back, front, position, detection):

		if back>=2:
			self.tracking.append({'track_id': detection['id'], 'detections':0, 'lastpose': position, 'speed':0, 'currentpose': position, 'computes': 0, 'orientation': 'away', 'lastframe':self.framecount, 'currentframe': self.framecount, 'latest':0})
			#print ("orientation", "away", back, front, position)
		elif front>=3:
			self.tracking.append({'track_id': detection['id'], 'detections':0, 'lastpose': position, 'speed':0, 'currentpose': position, 'computes': 0, 'orientation': 'towards', 'lastframe':self.framecount, 'currentframe': self.framecount, 'latest':0})
			#print ("orientation", "towards", back, front, position)
		else:
			self.tracking.append({'track_id': detection['id'], 'detections':0, 'lastpose': position, 'speed':0, 'currentpose': position, 'computes': 0, 'orientation': 'unknown', 'lastframe':self.framecount, 'currentframe': self.framecount, 'latest':0})
			#print ("orientation", "unknown", back, front, position)
	
	def init_frame(self, frame):

		self.prev_frame = frame

	def transform_points(self, points, class_box, frame):

		transformed_points=[]
		for point in points:
			x,y = int(point[0]+class_box['bbox'][0]), int(point[1]+class_box['bbox'][1])
			transformed_points.append([x,y])
			cv2.circle(frame, (x,y), 5, (0,0,255), -1)
		return frame, transformed_points

	def get_keypoints(self, auto_cropped):

		cropped = np.array(cv2.cvtColor(auto_cropped, cv2.COLOR_BGR2RGB))
		cropped_new = np.moveaxis(cropped, 2, 0)
		pointers, points = self.model.forward_pass(cropped_new)
		return pointers, points

	def check_orientation(self, pointers):

		back=0
		front=0

		for idx in pointers:
			if (idx == 0) or (idx == 1) or (idx == 6):
				back+=1
			elif (idx==17) or (idx==18) or (idx==19) or (idx==20) or (idx==21):
				front+=1

		return back, front

	def decrement(self, tracked_vehicle):

		tracked_vehicle['detections']-=1
		tracked_vehicle['computes']-=1
		return tracked_vehicle

	def update_tracked_frame(self, tracked_vehicle, position):

		tracked_vehicle['currentpose'] = position
		tracked_vehicle['currentframe'] = self.framecount
		tracked_vehicle['detections'] = 0
		return tracked_vehicle

	def update_speed(self, tracked_vehicle, position, orientation, speed):

		tracked_vehicle['speed'] += speed/100.0
		tracked_vehicle['lastpose'] = position
		#can be used for drawing boxes with speed info
		#extra, unnecessary = self.draw_boxes(frame, [i], tracked_vehicle['speed']/tracked_vehicle['computes'], True)
		print ("\n", 'speed', tracked_vehicle['speed']/tracked_vehicle['computes'], tracked_vehicle['track_id'], speed, orientation, tracked_vehicle['currentframe'], tracked_vehicle['lastframe'])
		tracked_vehicle['lastframe'] = self.framecount
		tracked_vehicle['latest'] = tracked_vehicle['speed']/tracked_vehicle['computes']

		return tracked_vehicle
		
	def speed_estimate(self, frame, class_boxes):
		
		#only need to call this function after init the class, see the arguments needed
		#returns tracking - a dictionary that contains vehicle ID, speed, number of computes, current and last pose (see the function "init_trackers" for a complete list of keywords)
		self.prev_frame = frame

		#i represents a single vehicle detection
		for i in class_boxes:
			
			if not(any(i['id'] == x for x in self.tracked)):

				frame, cropped = self.draw_boxes(frame, [i], 0, False)
				ht, wd = cropped.shape[0], cropped.shape[1]

				if (ht>100 and wd>100) or (ht>150) or (wd>150):
					
					pointers, points = self.get_keypoints(cropped)
					frame, transformed_points = self.transform_points(points, i, frame)
				
					if len(points)>=4:
						self.tracked.append(i['id'])
						position = self.compute_pose(pointers, transformed_points)
						back, front = self.check_orientation(pointers)
						self.init_trackers(back, front, position, i)

			elif (i['id'] == x for x in self.tracked):
				
				frame, cropped = self.draw_boxes(frame, [i], 0, False)
				ht, wd = cropped.shape[0], cropped.shape[1]

				for j in self.tracking:

					if j['track_id']==i['id']:
					
						frame, cray = self.draw_boxes(frame, [i], j['latest'], True)
						j['detections']+=1

						if j['detections']==2:
							print ("pair detected")# j["detections"], framecount)
							
							if (ht>100 and wd>100) or (ht>150) or (wd>150):
								
								j['computes']+=1
								pointers, points = self.get_keypoints(cropped)
								frame, transformed_points = self.transform_points(points, i, frame)
								#needed only to verify the number of keypoints detected
								#print ("length", len(points))

								if len(points)>=4:								
									position = self.compute_pose(pointers, transformed_points)
									
									if (j['orientation']=='away') and (position[2]>=j['lastpose'][2]):

										if ((abs(position[2]-j['lastpose'][2]))<(750+(self.framecount-j['lastframe'])*20)) and ((abs(position[2]-j['lastpose'][2]))>50): #(framecount-j['lastframe'])*20):

											j = self.update_tracked_frame(j, position)
											speed = self.distance(j['lastpose'], j['currentpose'])/(float(j['currentframe']-j['lastframe'])*self.time)
											if speed < 2000:
												j = self.update_speed(j, position, "moving away", speed)
											else:
												j = self.decrement(j)

										else:
											j = self.decrement(j)

									elif (j['orientation']=='towards') and (position[2]<=j['lastpose'][2]):
	
										if ((abs(position[2]-j['lastpose'][2]))<(750+(self.framecount-j['lastframe'])*20)) and ((abs(position[2]-j['lastpose'][2]))>50): #(framecount-j['lastframe'])*20):
											
											j = self.update_tracked_frame(j, position)
											speed = self.distance(j['lastpose'], j['currentpose'])/(float(j['currentframe']-j['lastframe'])*self.time)
											if speed<2000:
												#j = self.update_speed(j, speed, position, frame, [i], "moving closer", speed)												
												j = self.update_speed(j, position, "moving away", speed)
											else:
												j = self.decrement(j)
										else:
											j = self.decrement(j)

									elif (j['orientation']=='unknown'):

										back, front = self.check_orientation(pointers)
										
										if back>=2:
											j['orientation']='away'
										elif front>=3:
											j['orientation']='towards'

										if ((abs(position[2]-j['lastpose'][2]))<(750+(self.framecount-j['lastframe'])*20)) and ((abs(position[2]-j['lastpose'][2]))>50): #(framecount-j['lastframe'])*20):
						
											j = self.update_tracked_frame(j, position)
											speed = self.distance(j['lastpose'], j['currentpose'])/(float(j['currentframe']-j['lastframe'])*self.time)

											if speed<2000:
												j = self.update_speed(j, position, "moving away", speed)
												#j = self.update_speed(j, speed, position, frame, [i], "orientation unknown", speed)
											else:
												j = self.decrement(j)
										else:
											j = self.decrement(j)
									else:
										j = self.decrement(j)
								else:
									j = self.decrement(j)
							else:
								j['detections']-=1

		for x in self.tracked:
			if not(any(x == m['id'] for m in class_boxes)):
				self.tracked.remove(x)

		print ("tracked IDs", self.tracked)
		self.frame_number+=1
		#contains speed estimates as well - you can also return the frames on which the speed values are written
		return self.tracking