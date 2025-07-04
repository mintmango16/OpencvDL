# 오츄알고리즘으로 이진 영상의 임계값 찾기 

import cv2 as cv
import sys
import os
os.getcwd()
os.chdir("03_영상 처리")

img=cv.imread('soccer.jpg') 

# 최적 임계값과 이진화 영상 저장 
t, bin_img=cv.threshold(img[:,:,2], # 이진화할 채널 선택 
                       0,  #명암값 최소
                       255, # 최대 
                       cv.THRESH_BINARY + cv.THRESH_OTSU)#이진화 + 오츄 알고리즘 


print('오츄 알고리즘이 찾은 최적 임곗값=',t)
#오츄 알고리즘이 찾은 최적 임곗값= 113.0

cv.imshow('R channel',img[:,:,2])			# R 채널 영상
cv.imshow('R channel binarization',bin_img)	# R 채널 이진화 영상

cv.waitKey()
cv.destroyAllWindows()