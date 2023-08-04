import cv2
import numpy as np
import matplotlib.pyplot as plt

#Detect ball with Houghcircles

kernel_sharp = np.array([[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]])

for i in range(1, 7):

	img = cv2.imread(f"football/{i}.jpg")
	img_show1 = img.copy()
	img_show2 = img[175:266 , 265:315].copy()

	gray = cv2.cvtColor(img[175:266 , 265:315], cv2.COLOR_BGR2GRAY)

	sharp = cv2.filter2D(gray, -1, kernel_sharp)
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
	sharp2 = cv2.filter2D(thresh, -1, kernel_sharp)
	blur2 = cv2.GaussianBlur(thresh, (5, 5), 0)

	thresh = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
	# kernel = np.ones((3, 3), np.uint8)
	# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

	blur = cv2.GaussianBlur(thresh, (5, 5), 0)

	for i in range(0,15):
		blur = cv2.GaussianBlur(blur, (5, 5), 0)

	sharp3 = cv2.filter2D(blur, -1, kernel_sharp)
	sharp4 = cv2.filter2D(sharp3, -1, kernel_sharp)

	# cv2.imshow("Frames", sharp4)
	# cv2.waitKey(0)

	circles = cv2.HoughCircles(sharp4, cv2.HOUGH_GRADIENT, 3.9, 60, param1=30, param2=50, minRadius=6,maxRadius=12)

	if circles is not None:
		circles = circles[0].astype(np.uint32)

		for circle in circles:				
			cv2.circle(img_show1, (circle[0]+265, circle[1]+175), circle[2], (0, 0, 255), 2)

	cv2.imshow("Frames", img_show1)
	cv2.waitKey(0)

	# plt.imshow(img_show, cmap="gray")
	# plt.show()

cv2.destroyAllWindows()

#Detect line with HoughLines

import random

for i in range(1, 7):

	img = cv2.imread(f"football/{i}.jpg")
	img_show1 = img.copy()
	img_show2 = img[172:351 , 223:417].copy()

	gray = cv2.cvtColor(img_show2, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)

	for i in range(0,5):
		blur = cv2.GaussianBlur(blur, (5, 5), 0)

	thresh = cv2.adaptiveThreshold(blur, 270, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

	
	edges = cv2.Canny(blur, threshold1=5, threshold2=15, apertureSize=3)

	# cv2.imshow('Line Detection', edges)
	# cv2.waitKey(0)

	lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)

	if lines is not None:
	    for line in lines:
	        x1, y1, x2, y2 = line[0]
	        cv2.line(img_show1, (223+x1, 172+y1), (223+x2, 172+y2), (0, 0, 255), 2)  # Draw red line

	cv2.imshow('Line Detection', img_show1)
	cv2.waitKey(0)

cv2.destroyAllWindows()


#Detect ball with YOLO

from ultralytics import YOLO

model = YOLO("model/yolov8x")

for i in range(1, 7):
	img = cv2.imread(f"football/{i}.jpg")

	results = model.predict(source=img)

	for result in results:
		for (obj_xyxy, obj_cls) in zip(result.boxes.xyxy, result.boxes.cls):
			obj_cls = int(obj_cls)

			if obj_cls == 32:
				x1 = obj_xyxy[0].item()
				y1 = obj_xyxy[1].item()
				x2 = obj_xyxy[2].item()
				y2 = obj_xyxy[3].item()

				ball_x = (x2+x1)/2
				ball_y = (y2+y1)/2

				ball_r = ((x2-x1)/2+(y2-y1)/2)/2

				cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
		

	cv2.imshow("Frames", img)

	cv2.waitKey(0)

cv2.destroyAllWindows()