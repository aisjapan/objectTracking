#!/usr/local/bin/python
#! -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

#FrameSize(This Robot Use 800*600[pixel])
FRAME_W = 800
FRAME_H = 600

# Cut Mirror Area
def cut_circle(img):

    # Define 3Layer(BGR)
    size = FRAME_H, FRAME_W , 1

    # Fill in black use np.fill
    white_img = np.zeros(size, dtype=np.uint8)
    white_img.fill(0)

    # Draw Circle filled with White to the right in the Center.
    cv2.circle(white_img, ((int)(FRAME_W/2-10), (int)(FRAME_H/2+30)), 210, (255), -1)
    
    # Mask Processing(Cut Only Mirror Area)
    result = cv2.bitwise_and(img, img, mask=white_img)
    
    return result

# Template Matching
def template_matching(img_rgb,template,blocksize,param1,i):
  img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#  th3 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blocksize,param1)
  w, h = template.shape[::-1]

  res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
  threshold = 0.3
  loc = np.where( res >= threshold)
  for pt in zip(*loc[::-1]):
      cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
      print (h)
      #print (str((h*200)/87) + '[mm]')
  
  cv2.imwrite('C:\\Users\\nct20\\Documents\\GitHub\\objectTracking\\image_output\\template_test\\template_result' + '_' + str(i) + '.jpg',img_rgb)

if __name__ == '__main__':
  image = cv2.imread('C:\\Users\\nct20\\Documents\\GitHub\\objectTracking\\image_input\\300mm.jpg')
  template = cv2.imread('C:\\Users\\nct20\\Documents\\GitHub\\objectTracking\\image_input\\robot_sample_200mm.jpg',0)
  image_cut = cut_circle(image)
  for i in range(1,2):
      template_matching(image_cut,template,99,-10,i)