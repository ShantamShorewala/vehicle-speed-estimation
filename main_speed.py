"""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
"""
import cv2
#from deepsort import deepsort_rbc
from yolov3 import YOLOV3
import cv2
import os
import time
import numpy as np
from analytics import analytics_rbc
from speed_estimate_class import speed_estimation

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

def draw_boxes(frame,class_boxes, speed, onlyspeed):

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
		if (onlyspeed==True):
			cv2.putText(frame, "Speed:"+str(round(speed, 2))+ "m/s" , (x2+50,y2), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0),2, cv2.LINE_AA)
		#cv2.putText(frame, "CLASS:"+str(classname) , (x2,y2),cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0), 1, cv2.LINE_AA)
		#cv2.putText(frame, "Speed:"+ str(speed) +' Km/hr' , (x2+10,y2+10),cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0 , 0), 2, cv2.LINE_AA)
	return frame, cropped


cap = cv2.VideoCapture('6Mar.mp4')
#cap.set(cv2.CAP_PROP_FPS, 30)
print(cv2.CAP_PROP_FPS)

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('test_out_3.avi', fourcc, 15.0, (int(cap.get(3)), int(cap.get(4))))


#yolo = YOLOV3("cfg/yolo_2k_reanchored.cfg","weights/yolo_2k_reanchored_70000.weights","cfg/2k_aug.data")

count_line = [(650,400),(1000,500)]

#sline_1_203 = [(1065,514),(1038,704)]


#sline_1_203 = [(1065,514),(697,406)]
#sline_2_203 = [(1038,704),(412,506)]

sline_1_203 = [(1065,514),(524,366)]
sline_2_203 = [(1038,704),(151,406)]

a = analytics_rbc(count_line,sline_1_203,sline_2_203)
initial_det=0

while True:
	ret, frame = cap.read()	

	if frame is None:
		break
	
	if frame is not None:
		#cv2.imwrite('temp/image.jpg',frame)
		#out_scores, out_boxes, out_classes = yolo.detect_image(frame)
		'''counts,'''
		initial_det+=1

		if initial_det==1:
			print "test"
			speeder = speed_estimation(frame.shape[0], frame.shape[1])
			speeder.init_frame(frame)

		speeder.framecount+=1
		counts, class_boxes = a.run_analytics(frame,300)
		tracking = speeder.speed_estimate(frame, class_boxes)
		
#		print tracking
		
		
