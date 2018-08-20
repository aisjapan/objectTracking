#!/usr/local/bin/python
#! -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys

# フレームサイズ
FRAME_W = 800
FRAME_H = 600

GRAY_THRESHOLD = 50 # GRAYスケールの閾値

#コマンドライン引数の受け取りのため
args = sys.argv

# 指定した画像(path)の物体を検出し、外接矩形の画像を出力します
def detect_contour(src, blocksize, param1):
  
  # グレースケール画像へ変換
  gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

  # 2値化 元は50
  #retval, th1 = cv2.threshold(gray, GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)
  #th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
  th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blocksize,param1)
  #cv2.imshow('Result',th3)

  # 輪郭を抽出
  #   contours : [領域][Point No][0][x=0, y=1]
  #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
  #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
  image, contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # ObjectTrackingは、findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # 矩形検出された数（デフォルトで0を指定)
  detect_count = 0

  # 各輪郭に対する処理
  for i in range(0, len(contours)):

    # 輪郭の領域を計算
    area = cv2.contourArea(contours[i])

    # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
    if area < 1e2 or 1e5 < area:
      continue

    # 外接矩形
    if len(contours[i]) > 0:
      rect = contours[i]
      x, y, w, h = cv2.boundingRect(rect)
      cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

      # 外接矩形毎に画像を保存
      # cv2.imwrite('{C:\Users\nct20\Documents\AIS}' + str(detect_count) + '.jpg', src[y:y + h, x:x + w])

      detect_count = detect_count + 1

  # 外接矩形された画像を表示
  cv2.imshow('output', src)
  
  k = cv2.waitKey(0)
  # ESC:プログラム終了,s:セーブ＋プログラム終了
  if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
  elif k == ord('s'): # wait for 's' key to save and exit
   cv2.imwrite('C:\\Users\\nct20\\Documents\\GitHub\\objectTracking\\image_output\\200mm_test\\' + args[1] + '_' + str(blocksize) + '_' + str(int(param1)) + '.jpg',src)
   cv2.destroyAllWindows()

  # 終了処理
  cv2.destroyAllWindows()
  
# ミラー領域だけを切り取る
def cut_circle(img):

    # 3レイヤー(BGR)を定義
    size = FRAME_H, FRAME_W , 1

    # np.fillで黒に埋める
    white_img = np.zeros(size, dtype=np.uint8)
    white_img.fill(0)

    # 中央右寄りに、白で塗りつぶされた円形を描く
    cv2.circle(white_img, ((int)(FRAME_W/2), (int)(FRAME_H/2+10)), 250, (255), -1)
    
    # マスク処理(ミラー部分だけを切り取る)
    result = cv2.bitwise_and(img, img, mask=white_img)
    
    return result

if __name__ == '__main__':
  for i in range(1,2):
  	image = cv2.imread('C:\\Users\\nct20\\Documents\\GitHub\objectTracking\\image_input\\200mm.jpg')
  	image = cut_circle(image)
  	image = cv2.flip(image,1)
  	detect_contour(image,11,float(i))