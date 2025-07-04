# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 22:10:28 2025

@author: asiae
"""

import cv2 as cv
import sys

img = cv.imread('hangang.jpg')

if img is None:
    sys.exit('no file')
    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 명암 영상으로 변환
gray_small = cv.resize(gray, dsize=(0,0), fx = 0.5, fy = 0.5) # 반으로 축소 

cv.imwrite('hangang_gray.jpg', gray) # 영상 파일로 저장 
cv.imwrite('hangang_gray_small.jpg', gray_small)


cv.imshow('color image',img)
cv.imshow('gray image',gray)
cv.imshow('gray_small image',gray_small)

cv.waitKey()
cv.destroyAllWindows()

