# -*- coding: utf-8 -*-
# Moving Image + Recognition of Red Color + Display of Maximum Area
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import serial
import threading
import json


# フレームサイズ
FRAME_W = 800
FRAME_H = 600

HSV_CONV = 0.70 #HSV, 360dankai => 256dankai , 250/360 = 0.7
#colorRange = [[RedMin],[RedMax],[BlueMin],[BlueMax],[GreenMin],[GreenMax],[PurpleMin][PurpleMax],[YellowMin][YellowMax]]
colorRange = [[30*HSV_CONV,330*HSV_CONV], [210*HSV_CONV,240*HSV_CONV], [80*HSV_CONV,150*HSV_CONV], [240*HSV_CONV,300*HSV_CONV],[40*HSV_CONV,80*HSV_CONV]]


# Json
f = open('RaspberryPiMessage_A.json', 'r')
send_message_dict = json.load(f)

#Serial
#port = "/dev/ttyACM0"
#port = "COM4"
port = "/dev/ttyS0"
serialFromArduino = serial.Serial(port, 9600)
serialFromArduino.flushInput()
angle = 0

#-----------------------------------------------------------------------------------

#Serial Communication
def SendJson2Due():
    if angle < 180 and angle > 20:
        send_message_dict["movement"] = 3
    elif angle > 180 and angle < 340:
        send_message_dict["movement"] = 2
    else:
        send_message_dict["movement"] = 1
    send_str = json.dumps(send_message_dict)
    send_str = send_str + '\n'
    serialFromArduino.write(bytes(send_str.encode("utf-8")))
    # serialFromArduino.write(bytes('off\n', 'utf-8'))
    print("hello1")
    print(send_str)
    print("hello2")
    t=threading.Timer(1,SendJson2Due)
    t.start()

def find_rect_of_target_color(getImage,colorNum):
  hsv = cv2.cvtColor(getImage, cv2.COLOR_BGR2HSV_FULL)
  h = hsv[:, :, 0]
  s = hsv[:, :, 1]
  mask = np.zeros(h.shape, dtype=np.uint8)
  
  #Red
  if(colorNum == 1):
      mask[( (h < colorRange[0][0]) | (h > colorRange[0][1]) ) & (s > 128)] = 255
  #Blue
  elif(colorNum == 2):
      mask[((h > colorRange[1][0]) & (h < colorRange[1][1])) & (s > 128)] = 255
  #GReeeeN
  elif(colorNum == 3):
      mask[((h > colorRange[2][0]) & (h < colorRange[2][1])) & (s > 60)] = 255
  #Purple
  elif(colorNum == 4):
      mask[((h > colorRange[3][0]) & (h < colorRange[3][1])) & (s > 50)] = 255
  #Yellow
  elif(colorNum == 5):
      mask[((h > colorRange[4][0]) & (h < colorRange[4][1])) & (s > 128)] = 255

  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  rects = []
  
  for i in range(0,2):
      for contour in contours:
        approx = cv2.convexHull(contour)
        rect = cv2.boundingRect(approx)
        rects.append(np.array(rect))

  return rects

# ミラー領域だけを切り取る
def cut_circle(img):

    # 3レイヤー(BGR)を定義
    size = FRAME_H, FRAME_W , 1

    # np.fillで白に埋める
    white_img = np.zeros(size, dtype=np.uint8)
    white_img.fill(0)

    # 中央右寄りに、青で塗りつぶされた円形を描く
    cv2.circle(white_img, ((int)(FRAME_W/2+30), (int)(FRAME_H/2)), 330, (255, 255, 255), -1)

    # マスク処理(ミラー部分だけを切り取る)
    result = cv2.bitwise_and(img, img, mask=white_img)
	
    return result

def calcAngle(originX,originY,x,y):

    degree = np.rad2deg(np.arctan2((y-originY),(x-originX)))

    # 第1象限
    if x >= originX and y <= originY:
	degree = np.fabs(degree) +270
    # 第2象限
    elif x <= originX and y <= originY:
	degree = np.fabs(degree) - 90
    # 第3象限
    elif x <= originX and y >= originY:
	degree = 270 - np.fabs(degree)
    # 第4象限
    elif x>= originX and y >= originY:
	degree = 270 - np.fabs(degree)

    return degree

def calcDistance(originX,originY,x,y):

    distance = np.sqrt((x-originX) * (x-originX) + (y-originY) * (y-originY))

    return distance
    

if __name__ == '__main__':

  camera = PiCamera()
  camera.resolution = (FRAME_W, FRAME_H)
  camera.awb_mode = 'auto'
  camera.framerate = 32
  rawCapture = PiRGBArray(camera, size=(FRAME_W, FRAME_H))
 
  # allow the camera to warmup
  time.sleep(0.005)
 
  rectsR = []
  rectsB = []
  rectsG = []
  print ("Hello!!")
  
  t=threading.Thread(target=SendJson2Due)
  t.start()

  # capture frames from the camera
  for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  	# grab the raw NumPy array representing the image, then initialize the timestamp
  	# and occupied/unoccupied text
  	frame_image = frame.array
  	
  	image = cut_circle(frame_image)
  	image = cv2.flip(image,1)
#        image = frame_image	
	rectsR = find_rect_of_target_color(image, 1)
        rectsB = find_rect_of_target_color(image, 2)
        rectsG = find_rect_of_target_color(image, 3)
        rectsP = find_rect_of_target_color(image, 4)
        rectsY = find_rect_of_target_color(image, 5)
        if len(rectsR) >0:
            rectR = max(rectsR, key = (lambda x: x[2] * x[3]))
            cv2.rectangle(image, tuple(rectR[0:2]), tuple(rectR[0:2] + rectR[2:4]), (0, 0, 255), thickness=2)
        if len(rectsB) >0:
            rectB = max(rectsB, key = (lambda x: x[2] * x[3]))
            cv2.rectangle(image, tuple(rectB[0:2]), tuple(rectB[0:2] + rectB[2:4]), (255, 0, 0), thickness=2)
        if len(rectsG) >0:
            rectG = max(rectsG, key = (lambda x: x[2] * x[3]))
            cv2.rectangle(image, tuple(rectG[0:2]), tuple(rectG[0:2] + rectG[2:4]), (0, 255, 0), thickness=2)
        if len(rectsP) >0:
            rectP = max(rectsP, key = (lambda x: x[2] * x[3]))
            cv2.rectangle(image, tuple(rectP[0:2]), tuple(rectP[0:2] + rectP[2:4]), (168, 87, 167), thickness=2)
        if len(rectsY) >0:
            rectY = max(rectsY, key = (lambda x: x[2] * x[3]))
            cv2.rectangle(image, tuple(rectY[0:2]), tuple(rectY[0:2] + rectY[2:4]), (0, 199, 227), thickness=2)


        originX,originY = (int)(FRAME_W/2+10), (int)(FRAME_H/2-20)
        x, y = rectR[0]+(int)(rectR[2]/2.0), rectR[1]+(int)(rectR[3]/2.0)

	cv2.circle(image,(originX,originY), 20, (0,0,0), -1)

        # calc angle
        angle = calcAngle(originX,originY,x,y)
	print("Angle = ", angle)

        # calc distance
	distance = calcDistance(originX,originY,x,y)
	print("Distance = ", distance)
	

        cv2.imshow('Recognition Now!', image)
	
	# show the frame
	#cv2.imshow("Frame", gray)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)

 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	"""
            #Receiving Serial data
        input_str = serialFromArduino.readline()
        if(len(input_str)>0):
            json_str = input_str

            #string2json
            json_dict = json.loads(json_str)

            print(json_dict)

            #changing data
            send_message_dict["led"] = json_dict["led"][0] + 1
            send_message_dict["led"] %= 16
        """        
            
  serialFromArduino.close()