"""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
"""
from __future__ import print_function, absolute_import
import cv2
#from deepsort import deepsort_rbc
from yolov3 import YOLOV3
import cv2
import os
import time, math
import numpy as np
from auto_track import analytics_rbc
import hourglass_single#.example.hourglass_single #import hourglass

def distance(last_pos, current_pos):
	if last_pos[2]<0:
		last_pos*=-1
	if current_pos[2]<0:
		current_pos*=-1

	val=0
	for i in range(0,3):
		val += (last_pos[i]-current_pos[i])*(last_pos[i]-current_pos[i])

	val=math.pow(val,0.5)
	return val


def compute_pose(points, imagepoints):
	#pts={0: [49.0, 56.0, 116.5], 1: [-57.0, 57.8, 145.5], 2: [0.0, 0.0, 38.0], 3: [51.0, 240.0, 149.5], 4: [-33.0, 36.0, 83.5], 5: [45.0, 256.5, 71.0], 6: [66.4, 220.0, 19.8], 7: [47.5, 43.6, 104.7], 8: [-54.3, 135.0, 149.8], 9: [-43.0, 68.0, 165.6], 10: [-44.2, 61.5, 152.2], 11: [0.0, 260.0, 41.5], 12: [-47.5, 43.6, 104.7], 13: [-45.0, 256.5, 71.0], 14: [56.0, 186.7, 89.0], 15: [43.0, 68.0, 165.6], 16: [33.0, 36.0, 83.5], 17: [-66.4, 220.0, 19.8], 18: [57.0, 57.8, 145.5], 19: [54.3, 135.0, 149.8], 20: [44.2, 61.5, 152.4], 21: [-49.0, 56.0, 116.5], 22: [-51.0, 240.0, 149.5], 23: [-56.0, 186.7, 89.0], 24: [45.0, 125.0, 117.5]}

	pts={0: [45.0, 256.5, 71.0], 1: [-45.0, 256.5, 71.0], 2: [56.0, 186.7, 89.0], 3: [-56.0, 186.7, 89.0], 4: [51.0, 240.0, 149.5], 5: [-51.0, 240.0, 149.5], 6: [0.0, 260.0, 41.5], 7: [-47.5, 43.6, 104.7], 8: [-57.0, 57.8, 145.5], 9: [-54.3, 135.0, 149.8], 10: [-66.4, 220.0, 19.8], 11: [-43.0, 68.0, 165.6], 12: [57.0, 57.8, 145.5], 13: [54.3, 135.0, 149.8], 14: [43.0, 68.0, 165.6], 15: [47.5, 43.6, 104.7], 16: [66.4, 220.0, 19.8], 17: [44.2, 61.5, 152.4], 18: [-44.2, 61.5, 152.2], 19: [49.0, 56.0, 116.5], 20: [-49.0, 56.0, 116.5], 21: [0.0, 0.0, 38.0], 22: [33.0, 36.0, 83.5], 23: [-33.0, 36.0, 83.5]}


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

	#print ("passing", twist, worldpoints)
	_, rvecs, tvecs = cv2.solvePnP(worldpoints, twist, mtx, dist)
	
	return tvecs

def format_save(out_boxes,out_classes,frame):
	for i in range(len(out_boxes)):
		x,y,w,h = out_boxes[i]
		cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
		cv2.putText(frame, str(out_classes[i].decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
		out.write(frame)

def draw_counts(frame,counts):

	#counts['up_count']
	#counts['down_count']

	up_count_dict = {k:counts['up_count'][k] for k in sorted(counts['up_count'])}
	down_count_dict = {k:counts['down_count'][k] for k in sorted(counts['down_count'])}
	classes = list(up_count_dict.keys())
	#print("Speed\n",speed)


	up_counts = list(up_count_dict.values())
	down_counts = list(down_count_dict.values())

	cv2.putText(frame, "Dn Count: " + str(up_counts), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 0), 2, cv2.LINE_AA)
	cv2.putText(frame, "Up Count: " + str(down_counts), (10, 80), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(frame, "Classes: " +str(classes), (10, 150),cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 0 , 0), 2, cv2.LINE_AA)

	return frame

def draw_boxes(frame,class_boxes):

	for item in class_boxes:
		classname = item['classname']
		bbox = item['bbox']
		id_number = item['id']
		#speed = item['speed']

		#Convert to Kilometer per hour and round to next integer
		#speed = speed * 3.6
		#speed = round(speed)

		bbox = list(map(int,bbox))

		x1,y1,x2,y2 = bbox
		#print ("weell", x1, x2, y1, y2)
		#time.sleep(1)

		#cv2.line(frame,count_line[0],count_line[1],(0,0,0),2)
		cropped = frame[y1:y2, x1:x2]
		cv2.rectangle(frame, (x1,y1),(x2,y2),(255, 255, 0),3)
		cv2.putText(frame, "ID:"+str(id_number) , (x2,y2),cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0),2, cv2.LINE_AA)
		cv2.putText(frame, str(classname.decode("utf-8")), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX,1.15, (0, 255, 255),2)
		#cv2.putText(frame, "CLASS:"+str(classname) , (x2,y2),cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0), 1, cv2.LINE_AA)
		#cv2.putText(frame, "Speed:"+ str(speed) +' Km/hr' , (x2+10,y2+10),cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0), 2, cv2.LINE_AA)
	return frame, cropped


cap = cv2.VideoCapture('/home/shantam/Documents/Programs/PoseEstimation/autospeed/clip 3/clip_3.avi')
#cap.set(cv2.CAP_PROP_FPS, 30)
print("fps rate: ", cv2.CAP_PROP_FPS)

#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('test_out_4.avi', fourcc, 15.0, (int(cap.get(3)), int(cap.get(4))))


#yolo = YOLOV3("cfg/yolo_2k_reanchored.cfg","weights/yolo_2k_reanchored_70000.weights","cfg/2k_aug.data")

count_line = [(650,400),(1000,500)]

#sline_1_203 = [(1065,514),(1038,704)]


#sline_1_203 = [(1065,514),(697,406)]
#sline_2_203 = [(1038,704),(412,506)]

sline_1_203 = [(1065,514),(524,366)]
sline_2_203 = [(1038,704),(151,406)]


a = analytics_rbc(count_line,sline_1_203,sline_2_203)
model = hourglass_single.Hourglass()

tracked = []
last = []
tracking = []

frame_number = 0

while True:
	global tracked, last, tracking
	found = 0

	if frame_number==0:
		ret, frame = cap.read()
		prev_frame = frame.copy()

	else:
		prev_frame = frame
		ret, frame = cap.read()


	if frame is None:
		break

	if frame is not None:
		#cv2.imwrite('temp/image.jpg',frame)
		#out_scores, out_boxes, out_classes = yolo.detect_image(frame)
		class_boxes = a.run_analytics(frame,300)

		for i in class_boxes:
			if not(any(i['id'] == x  for x in tracked)):
				frame, cropped = draw_boxes(frame, [i])
				#cropped = cv2.imread('/home/shantam/Documents/Programs/hourglasstensorlfow/images/cropped0.jpg')
				ht, wd = cropped.shape[0], cropped.shape[1]
				if ht>100 and wd>100:
					cropped = np.array(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
					#cropped = cv2.resize(cropped, (256,256))
					cropped_new = np.moveaxis(cropped, 2, 0)

					pointers, points = model.forward_pass(cropped_new)
					#print ("printing", pointers, points)

					for point in points:
						x,y = point[0], point[1]
						cv2.circle(cropped, (x,y), 5, (0,0,255), -1)
						#print (x,y)

					if len(points)>=4:
						tracked.append(i['id'])
						position = compute_pose(pointers, points)
						tracking.append({'track_id': i['id'], 'detections':0, 'lastpose': position, 'speed':0, 'currentpose': position, 'computes':0})
						cv2.imshow("initial", cropped)
						cv2.waitKey(10)
						
			elif (i['id'] == x for x in tracked):
				frame, cropped = draw_boxes(frame, [i])
				ht, wd = cropped.shape[0], cropped.shape[1]

				for j in tracking:
					if j['track_id']==i['id']:
						j['detections']+=1

						if j['detections']==5:
							if ht>100 and wd>100:
								j['computes']+=1
								cropped = np.array(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
								cropped_new = np.moveaxis(cropped, 2, 0)
								pointers, points = model.forward_pass(cropped_new)
								#print("points found", pointers, points)

								for point in points:
									print (point)
									x,y = point[0], point[1]
									cv2.circle(cropped, (x,y), 5, (0,0,255), -1)
									#print (x,y)

								cv2.imshow("initial", cropped)
								cv2.waitKey(10)

								if len(points)>=4:								
									position = compute_pose(pointers, points)
									j['currentpose'] = position
									j['detections'] = 0
									print ("poses here", j['lastpose'], j['currentpose'])

									speed = distance(j['lastpose'], j['currentpose'])
									j['speed'] += speed/100.0
									j['lastpose'] = position
									print ("\n", 'speed', j['speed']/j['computes'],  j['track_id'], len(points), speed)
									time.sleep(5)

								else:
									j['detections']-=1
									j['computes']-=1
							else:
								j['detections']-=1


		for x in tracked:
			if not(any(x == m['id'] for m in class_boxes)):
				tracked.remove(x)

		print ("so far", tracked)
		#time.sleep(.2)
		#for k in tracked:


		#frame = draw_boxes(frame,class_boxes)
		#frame = draw_counts(frame,counts)
		im = cv2.resize(frame,(740,580))
		if frame_number!=0:
			cv2.imshow('counts',frame)
			cv2.waitKey(10)
		#out.write(frame)
		#format_save(out_boxes,out_classes,frame)
		frame_number+=1