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
import sys
import math

# フレームサイズ
FRAME_W = 800
FRAME_H = 600

#読み込むピクチャのファイル名
FILE_DIST = 200

HSV_CONV = 0.70 #HSV, 360dankai => 256dankai , 250/360 = 0.7
#colorRange = [[RedMin],[RedMax],[BlueMin],[BlueMax],[GreenMin],[GreenMax],[PurpleMin][PurpleMax],[YellowMin][YellowMax]]
colorRange = [[30*HSV_CONV,330*HSV_CONV], [200*HSV_CONV,240*HSV_CONV], [80*HSV_CONV,150*HSV_CONV], [240*HSV_CONV,300*HSV_CONV],[40*HSV_CONV,80*HSV_CONV]]

# 距離導出関数のマジックナンバー (係数,指数)
COEFFICIENT_DF = 200198
INDEX_DF = -0.808

#コマンドライン引数の受け取りのため
args = sys.argv
#-----------------------------------------------------------------------------------
# Loading Image's
def loading_still_image():
  image = cv2.imread('C:\\Users\\nct20\\Documents\\GitHub\\objectTracking\\image_input\\'+str(FILE_DIST)+'mm.jpg')
  return image

#キー処理＆画面終了＆保存
def key_action(image):

  k = cv2.waitKey(0)
  # ESC:プログラム終了,s:セーブ＋プログラム終了
  if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
  elif k == ord('s'): # wait for 's' key to save and exit
   cv2.imwrite('C:\\Users\\nct20\\Documents\\GitHub\\objectTracking\\image_output\\' + args[1] + '.jpg',image)
   cv2.destroyAllWindows()

  # 終了処理
  cv2.destroyAllWindows()

# ミラー領域だけを切り取る
def cut_circle(img):

  # 3レイヤー(BGR)を定義
  size = FRAME_H, FRAME_W , 1

  # np.fillで白に埋める
  white_img = np.zeros(size, dtype=np.uint8)
  white_img.fill(0)

  # 中央右寄りに、青で塗りつぶされた円形を描く
  cv2.circle(white_img, ((int)(FRAME_W/2), (int)(FRAME_H/2+10)), 240, (255, 255, 255), -1)

  # マスク処理(ミラー部分だけを切り取る)
  result = cv2.bitwise_and(img, img, mask=white_img)

  return result

# Find Target Color
def find_rect_of_target_color(getImage, colorNum):
  hsv = cv2.cvtColor(getImage, cv2.COLOR_BGR2HSV_FULL)
  h = hsv[:, :, 0]
  s = hsv[:, :, 1]
  mask = np.zeros(h.shape, dtype=np.uint8)

  #Red
  if(colorNum == 1):
      mask[( (h < colorRange[0][0]) | (h > colorRange[0][1]) ) & (s > 128)] = 255
  #Blue
  elif(colorNum == 2):
      mask[((h > colorRange[1][0]) & (h < colorRange[1][1])) & (s > 60)] = 255
  #GReeeeN
  elif(colorNum == 3):
      mask[((h > colorRange[2][0]) & (h < colorRange[2][1])) & (s > 60)] = 255
  #Purple
  elif(colorNum == 4):
      mask[((h > colorRange[3][0]) & (h < colorRange[3][1])) & (s > 50)] = 255
  #Yellow
  elif(colorNum == 5):
      mask[((h > colorRange[4][0]) & (h < colorRange[4][1])) & (s > 128)] = 255

  getImage,contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  rects = []

  for i in range(0,2):
    for contour in contours:
      approx = cv2.convexHull(contour)
      rect = cv2.minAreaRect(approx) #回転を考慮した矩形の導出
      rects.append(np.array(rect))

  return rects

def draw_rotation_rectangle(rects,image):
  #rects [[0],[1]] [[0],[1]] = [[x],[y]] [[w],[h]]
  rect = tuple(max(rects, key = (lambda x: x[1][0] * x[1][1])))
  box = cv2.boxPoints(rect)
  box = np.int0(box)

  image = cv2.drawContours(image,[box],0,(0,0,255),2)
  print ('w:' + str(rect[1][0]) + ',h:' + str(rect[1][1]))
  print (str(COEFFICIENT_DF*((rect[1][0]*rect[1][1])**(INDEX_DF))))

if __name__ == '__main__':
  # Loading Still Image
  #image = loading_still_image()

  # Loading Moving Image
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

  # capture frames from the camera
  for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  	# grab the raw NumPy array representing the image, then initialize the timestamp
  	# and occupied/unoccupied text
  	frame_image = frame.array

    # ミラー領域の切り取り
  	image = cut_circle(frame_image)
    # 色領域の検出と矩形の値導出
  	rectsR = find_rect_of_target_color(image, 1)
  	rectsB = find_rect_of_target_color(image, 2)
  	rectsG = find_rect_of_target_color(image, 3)

  	if len(rectsB) >0:
  	  	draw_rotation_rectangle(rectsB,image)

  	cv2.imshow('Recognition Now.', image)
  	key_action(image)
