# -*- coding: utf-8 -*-
# Moving Image + Recognition of Red Color + Display of Maximum Area
# import the necessary packages

import time
import cv2
import numpy as np
import serial
import threading
import json
import sys
import math

#コマンドライン引数の受け取りのため
args = sys.argv

# フレームサイズ
FRAME_W = 800
FRAME_H = 600

#読み込むピクチャのファイル名
FILE_DIST = 500

HSV_CONV = 0.70 #HSV, 360dankai => 256dankai , 250/360 = 0.7
#colorRange = [[RedMin],[RedMax],[BlueMin],[BlueMax],[GreenMin],[GreenMax],[PurpleMin][PurpleMax],[YellowMin][YellowMax]]
colorRange = [[30*HSV_CONV,330*HSV_CONV], [200*HSV_CONV,240*HSV_CONV], [80*HSV_CONV,150*HSV_CONV], [240*HSV_CONV,300*HSV_CONV],[40*HSV_CONV,80*HSV_CONV]]

#-----------------------------------------------------------------------------------
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
      #rect = cv2.boundingRect(approx)

      rect = cv2.minAreaRect(approx)
      #box = cv2.boxPoints(rect)
      #box = np.int0(box)
      rects.append(np.array(rect))
      #box.append(np.array(box))

  return rects

if __name__ == '__main__':

  # Loading Image's
  image = cv2.imread('C:\\Users\\nct20\\Documents\\GitHub\\objectTracking\\image_input\\'+str(FILE_DIST)+'mm.jpg')
  template = cv2.imread('C:\\Users\\nct20\\Documents\\GitHub\\objectTracking\\image_input\\robot_sample_200mm.jpg',0)

  rectsR = []
  rectsB = []
  rectsG = []
  #print ("Hello!!")
  image = cut_circle(image)
  #image = cv2.flip(image,1)
  #image = frame_image

  rectsR = find_rect_of_target_color(image, 1)
  rectsB = find_rect_of_target_color(image, 2)
  rectsG = find_rect_of_target_color(image, 3)
  rectsP = find_rect_of_target_color(image, 4)
  rectsY = find_rect_of_target_color(image, 5)

  if len(rectsR) >0:
     rectR = max(rectsR, key = (lambda x: x[2] * x[3]))
     cv2.rectangle(image, tuple(rectR[0:2]), tuple(rectR[0:2] + rectR[2:4]), (0, 0, 255), thickness=2)
  if len(rectsB) >0:
     rectB = tuple(max(rectsB, key = (lambda x: x[1][0] * x[1][1])))
     #cv2.rectangle(image, tuple(rectB[0:2]), tuple(rectB[0:2] + rectB[2:4]), (255, 0, 0), thickness=1)
     #print (rectB)
     box = cv2.boxPoints(rectB)
     box = np.int0(box)
     image = cv2.drawContours(image,[box],0,(0,0,255),2)
     print ('w:' + str(rectB[1][0]) + ',h:' + str(rectB[1][1]))
     #print (str(900 - rectB[3]*700/42) + '[mm]')
     print (str(880141/(rectB[1][0]*rectB[1][1])))
     #print (str(880141/87.908*99.936*100/(rectB[1][0]*rectB[1][1])) + '[mm]')
     #ミラー画像上での距離の導出
     #print (math.sqrt( ((rectB[0][0] + rectB[1][0]/2) - FRAME_W/2)**2 + ((rectB[0][1] + rectB[1][1]/2) - (FRAME_H/2+10))**2 ))
     #radiusV = (math.sqrt( (rectB[0][0] - FRAME_W/2)**2 + (rectB[0][1] - (FRAME_H/2+10))**2 ))
     #print (str(radiusV/194.28394169359444 * 200) + '[mm]')

  #originX,originY = (int)(FRAME_W/2+10), (int)(FRAME_H/2-20)
  #x, y = rectR[0]+(int)(rectR[2]/2.0), rectR[1]+(int)(rectR[3]/2.0)
  #cv2.circle(image,(originX,originY), 20, (0,0,0), -1)

  # calc angle
  #angle = calcAngle(originX,originY,x,y)
  #print("Angle = ", angle)
  # calc distance
  #distance = calcDistance(originX,originY,x,y)
  #print("Distance = ", distance)

  cv2.imshow('Recognition Now.', image)
  key_action(image)
