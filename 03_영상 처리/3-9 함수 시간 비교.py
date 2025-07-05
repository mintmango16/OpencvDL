# 직접 작성한 함수와 OpenCV의 함수 시간 비교 

import cv2 as cv
import numpy as np
import time

import os
os.getcwd()
os.chdir("03_영상 처리")

def my_cvtGray1(bgr_img): # 모든 화소에 접근해 컬러 -> 명암 변환 
    g=np.zeros([bgr_img.shape[0],bgr_img.shape[1]])
    for r in range(bgr_img.shape[0]):
        for c in range(bgr_img.shape[1]):
            g[r,c]=0.114*bgr_img[r,c,0]+0.587*bgr_img[r,c,1]+0.299*bgr_img[r,c,2]
    return np.uint8(g)

def my_cvtGray2(bgr_img): # 파이썬의 배열 연산 구현 
    g=np.zeros([bgr_img.shape[0],bgr_img.shape[1]])
    g=0.114*bgr_img[:,:,0]+0.587*bgr_img[:,:,1]+0.299*bgr_img[:,:,2]
    return np.uint8(g)
    
img=cv.imread('rose.png') 

start=time.time()
my_cvtGray1(img)
print('My time1:',time.time()-start)

start=time.time()
my_cvtGray2(img)
print('My time2:',time.time()-start)

start=time.time()
cv.cvtColor(img,cv.COLOR_BGR2GRAY)
print('OpenCV time:',time.time()-start)

# 결과 
# My time1: 2.2071683406829834
# My time2: 0.00827789306640625
# OpenCV time: 0.004915475845336914

