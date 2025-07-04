# RGB 영상의 일부를 잘라내고 RGB 채널 분리 

import cv2 as cv
import sys
import os
os.getcwd()
os.chdir("03_영상 처리")

img=cv.imread('soccer.jpg') 
  
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
cv.imshow('original_RGB',img) # 원래 영상 디스플레이

cv.imshow('Upper left half',
          img[0 : img.shape[0]//2, # 세로 방향 절반 
              0 : img.shape[1]//2, # 가로 방향 절반 
              :])  # 모든 채널 (R,G,B)
 # 아래 -> 오른쪽 방향이기 때문에 왼쪽 위 1/4 부분만 호출  

cv.imshow('Center half',
          img[img.shape[0]//4 : 3*img.shape[0]//4,
              img.shape[1]//4 : 3*img.shape[1]//4,
              :])
# 첫번째/두번째 축 1/4~3/4까지 지정 -> 중간 부분만 호출 

cv.imshow('R channel',img[:,:,2])
cv.imshow('G channel',img[:,:,1])
cv.imshow('B channel',img[:,:,0])

cv.waitKey(0)
cv.destroyAllWindows()

# 각 채널은 단일 색생 강도를 담고 있는 grayscale image임 
# 각 채널만 따로 출력하면 2차원 배열(높이*배열)-> 그레이스케일 이미지로 출력함
# -> 채널 강도에 따라 흑백으로 보이는 것 
#----------------------------
# red 채널만 color로 보이게 하기 
import numpy as np
import cv2 as cv
import sys
import os
os.getcwd()
os.chdir("03_영상 처리")

img=cv.imread('soccer.jpg') 
zeros = np.zeros_like(img[:, :, 0]) # 이미지 하나의 채널과 같은 크기를 가진 2차원 영행렬 
red_only = cv.merge([zeros, zeros, img[:, :, 2]])  # B=0, G=0, R=있음

cv.imshow('Red in color', red_only)

cv.waitKey(0)
cv.destroyAllWindows()

