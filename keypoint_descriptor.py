import time, os, cv2, json

file = open("./annotations/auto_dataset_1010.txt")
file2 = open("./annotations/auto_dataset_546.txt")
file3 = open("./annotations/auto_dataset_190.txt")
#file4 = open("./annotations/auto_data_test_2.txt")

file5 = open("./annotations/auto_dataset_hflip_1010.txt")
file6 = open("./annotations/auto_dataset_hflip_546.txt")
file7 = open("./annotations/auto_dataset_hflip_190.txt")
#file8 = open("./annotations/auto_data_test_hflip_2.txt")

img_folder = "/home/shantam/Documents/Programs/hourglasstensorlfow/images"

array = []
array_test = []

count=0
for a in file:
	count+=1
	b = a.split(" ")
	b[0] = b[0][:len(b[0])]
	img_path = os.path.join(img_folder, b[0])
	print (img_path)
	img = cv2.imread(img_path)
	height, width = float(img.shape[0]), float(img.shape[1])	

	joints=[]
	keypoints=[]
	for i in range(5, 53, 2):
		x = [float(b[i]), float(b[i+1]), 1.00]
		if x[0]==-1.0:
			x[2] = 0.00

		else:
			keypoints.append([float(b[i]), float(b[i+1])])
		joints.append(x)
		
	
	print joints	
	
	# if (count>450 and count<471) or (count>950 and count<971):#count<30 or count>950 or (count > 200 and count <300):
	# 	array.append({"dataset": "MPI","isValidation": 1.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
	# 		"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})
	# else:
	# 	array.append({"dataset": "MPI","isValidation": 0.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
	# 		"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})


# count=0
# for a in file2:
# 	count+=1
# 	b = a.split(" ")
# 	b[0] = b[0][:len(b[0])]
# 	img_path = os.path.join(img_folder, b[0])
# 	print (img_path)
# 	img = cv2.imread(img_path)
# 	height, width = float(img.shape[0]), float(img.shape[1])	
# 	#print (height, width)
# 	#time.sleep(1)
# 	element["img_paths"] = b[0]
# 	element["img_height"] = height
# 	element["img_width"] = width
# 	scale = float(height)/200.0
# 	element["scale_provided"] = scale
# 	element["objpos"] = [height/2.0, width/2.0]

# 	joints=[]
# 	for i in range(5, 53, 2):
# 		x = [float(b[i]), float(b[i+1]), 1.00]
# 		if x[0]==-1.0:
# 			x[2] = 0.00
# 		joints.append(x)
	
# 	element['joint_self'] = joints
	
# 	if (count>450 and count<471) or (count>250 and count<271):#count<30 or count>950 or (count > 200 and count <300):
# 		array.append({"dataset": "MPI","isValidation": 1.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})
# 	else:
# 		array.append({"dataset": "MPI","isValidation": 0.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})


# count=0
# for a in file3:
# 	count+=1
# 	b = a.split(" ")
# 	b[0] = b[0][:len(b[0])]
# 	img_path = os.path.join(img_folder, b[0])
# 	print (img_path)
# 	img = cv2.imread(img_path)
# 	height, width = float(img.shape[0]), float(img.shape[1])	
# 	#print (height, width)
# 	#time.sleep(1)
# 	element["img_paths"] = b[0]
# 	element["img_height"] = height
# 	element["img_width"] = width
# 	scale = float(height)/200.0
# 	element["scale_provided"] = scale
# 	element["objpos"] = [height/2.0, width/2.0]

# 	joints=[]
# 	for i in range(5, 53, 2):
# 		x = [float(b[i]), float(b[i+1]), 1.00]
# 		if x[0]==-1.0:
# 			x[2] = 0.00
# 		joints.append(x)
	
# 	element['joint_self'] = joints
	
# 	if (count>50 and count<71) or (count>150 and count<171):#count<30 or count>950 or (count > 200 and count <300):
# 		array.append({"dataset": "MPI","isValidation": 1.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})
# 	else:
# 		array.append({"dataset": "MPI","isValidation": 0.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})


# count=0
# for a in file5:
# 	count+=1
# 	b = a.split(" ")
# 	b[0] = b[0][:len(b[0])]
# 	img_path = os.path.join(img_folder, b[0])
# 	print (img_path)
# 	img = cv2.imread(img_path)
# 	height, width = float(img.shape[0]), float(img.shape[1])	
# 	#print (height, width)
# 	#time.sleep(1)
# 	element["img_paths"] = b[0]
# 	element["img_height"] = height
# 	element["img_width"] = width
# 	scale = float(height)/200.0
# 	element["scale_provided"] = scale
# 	element["objpos"] = [height/2.0, width/2.0]

# 	joints=[]
# 	for i in range(5, 53, 2):
# 		x = [float(b[i]), float(b[i+1]), 1.00]
# 		if x[0]==-1.0:
# 			x[2] = 0.00
# 		joints.append(x)
	
# 	element['joint_self'] = joints
	
# 	if (count>450 and count<471) or (count>950 and count<971):#count<30 or count>950 or (count > 200 and count <300):
# 		array.append({"dataset": "MPI","isValidation": 1.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})
# 	else:
# 		array.append({"dataset": "MPI","isValidation": 0.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})



# count=0
# for a in file6:
# 	count+=1
# 	b = a.split(" ")
# 	b[0] = b[0][:len(b[0])]
# 	img_path = os.path.join(img_folder, b[0])
# 	print (img_path)
# 	img = cv2.imread(img_path)
# 	height, width = float(img.shape[0]), float(img.shape[1])	
# 	#print (height, width)
# 	#time.sleep(1)
# 	element["img_paths"] = b[0]
# 	element["img_height"] = height
# 	element["img_width"] = width
# 	scale = float(height)/200.0
# 	element["scale_provided"] = scale
# 	element["objpos"] = [height/2.0, width/2.0]

# 	joints=[]
# 	for i in range(5, 53, 2):
# 		x = [float(b[i]), float(b[i+1]), 1.00]
# 		if x[0]==-1.0:
# 			x[2] = 0.00
# 		joints.append(x)
	
# 	element['joint_self'] = joints
	
# 	if (count>50 and count<71) or (count>250 and count<271):#count<30 or count>950 or (count > 200 and count <300):
# 		array.append({"dataset": "MPI","isValidation": 1.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})
# 	else:
# 		array.append({"dataset": "MPI","isValidation": 0.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})

# count=0
# for a in file7:
# 	count+=1
# 	b = a.split(" ")
# 	b[0] = b[0][:len(b[0])]
# 	img_path = os.path.join(img_folder, b[0])
# 	print (img_path)
# 	img = cv2.imread(img_path)
# 	height, width = float(img.shape[0]), float(img.shape[1])	
# 	#print (height, width)
# 	#time.sleep(1)
# 	element["img_paths"] = b[0]
# 	element["img_height"] = height
# 	element["img_width"] = width
# 	scale = float(height)/200.0
# 	element["scale_provided"] = scale
# 	element["objpos"] = [height/2.0, width/2.0]

# 	joints=[]
# 	for i in range(5, 53, 2):
# 		x = [float(b[i]), float(b[i+1]), 1.00]
# 		if x[0]==-1.0:
# 			x[2] = 0.00
# 		joints.append(x)
	
# 	element['joint_self'] = joints
	
# 	if (count>50 and count<71) or (count>150 and count<171):#count<30 or count>950 or (count > 200 and count <300):
# 		array.append({"dataset": "MPI","isValidation": 1.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})
# 	else:
# 		array.append({"dataset": "MPI","isValidation": 0.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})

# # count=0
# # for a in file:
# # 	count+=1
# # 	b = a.split(" ")
# # 	b[0] = b[0][:len(b[0])-1]
# # 	img_path = os.path.join(img_folder, b[0])
# # 	print (img_path)
# # 	img = cv2.imread(img_path)
# # 	height, width = float(img.shape[0]), float(img.shape[1])	
# # 	#print (height, width)
# # 	#time.sleep(1)
# # 	element["img_paths"] = b[0]
# # 	element["img_height"] = height
# # 	element["img_width"] = width
# # 	scale = float(height)/200.0
# # 	element["scale_provided"] = scale
# # 	element["objpos"] = [height/2.0, width/2.0]

# # 	joints=[]
# # 	for i in range(5, 53, 2):
# # 		x = [float(b[i]), float(b[i+1]), 1.00]
# # 		if x[0]==-1.0:
# # 			x[2] = 0.00
# # 		joints.append(x)
	
# # 	element['joint_self'] = joints
	
# # 	if count<30 or count>950 or (count > 200 and count <300):
# # 		array.append({"dataset": "MPI","isValidation": 1.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# # 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})
# # 	else:
# # 		array.append({"dataset": "MPI","isValidation": 0.000,"img_paths": b[0],"img_width": width,"img_height": height,"objpos": [height/2.0, width/2.0],
# # 			"joint_self": joints,"scale_provided": scale,"annolist_index": 5.000,"people_index": 1.000,"numOtherPeople": 1.000})

# print len(array), len(array_test)
# #time.sleep(5)

# with open('auto_augmented_train_mpii.json', 'w') as outfile:  
#     json.dump(array, outfile)
    
# for i in array:
# 	print i
# 	time.sleep(1)
