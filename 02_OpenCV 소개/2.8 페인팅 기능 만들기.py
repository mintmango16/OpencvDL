# -*- coding: utf-8 -*-

import cv2 as cv
import sys

img = cv.imread('hangang.jpg')

if img is None :
    sys.exit('no file')
    
BrushSiz= 5
LColor, RColor = (255, 0, 0), (0, 0, 255)

def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), BrushSiz, LColor, -1)
    elif event ==cv.EVENT_RBUTTONDOWN:
        cv.circle(img, (x, y), BrushSiz, RColor, -1)
    elif event ==cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x, y), BrushSiz, LColor, -1)
    elif event ==cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img, (x, y), BrushSiz, RColor, -1)
        
    cv.imshow('Painting', img)

cv.namedWindow('painting')
cv.imshow('Painting', img)

cv.setMouseCallback('Painting', painting)

while True:
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break
