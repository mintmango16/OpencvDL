import cv2 as cv
import numpy as np

import os
os.getcwd()
os.chdir("03_영상 처리")

img=cv.imread('soccer.jpg') 
img=cv.resize(img,dsize=(0,0),fx=0.25,fy=0.25) # 1/4로 축소 

# 감마 함수 정의 
def gamma(f,gamma=1.0):
    """f : 감마 보정 대상 영상,
    gamma : 감마 보정 식의 gamma=밝기 변화 조정
    >1은 어두워짐, <1은 밝아짐, =1은 밝기 유지"""
    f1=f/255.0			# L=256이라고 가정
    return np.uint8(255*(f1**gamma))

 # 감마 조정하여 모두 이미지 이어붙이기 
gc=np.hstack((gamma(img,0.5),
              gamma(img,0.75),
              gamma(img,1.0), # 원본 밝기 유지 
              gamma(img,2.0),
              gamma(img,3.0)))
cv.imshow('gamma', gc)

cv.waitKey()
cv.destroyAllWindows()