# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 22:35:45 2025

@author: asiae
"""

import cv2 as cv 
import sys

img = cv.imread('hangang.jpg')

if img is None :
    sys.exit('no file')
    
    
# cv.rectangle(img, (400, 30), (600, 200), (0, 0, 255), 2) # 직사각형 삽입
# # 영상, 왼쪽위 좌표(x,y), 오른쪽 아래 좌표, 색 지정 BGR, 선의 두께 

# cv.putText(img, 'light', (450, 24), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # 글씨 삽입
# # 좌하단 좌표, 폰트, 글자 크기, 색, 글자 두께 
# cv.imshow('Draw', img)

# cv.waitKey()
# cv.destroyAllWindows()

def draw(event, x, y, flags, param):
    global ix, iy
    
    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 클릭
        #cv.rectangle(img, (x, y), (x+100, y+100), (0, 0, 255), 2)
        ix, iy = x, y
    elif event == cv.EVENT_RBUTTONDOWN: 
        cv.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)
        
    cv.imshow('Drawing', img)
    
cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw)

while True:
    if cv.waitKey(1) == ord('q'):
        cv.destoryAllWindows()
        break