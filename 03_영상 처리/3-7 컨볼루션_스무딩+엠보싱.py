import cv2 as cv
import numpy as np

import os
os.getcwd()
os.chdir("03_영상 처리")

img=cv.imread('soccer.jpg')
img=cv.resize(img,dsize=(0,0),fx=0.4,fy=0.4)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 명암 영상 변환 : np.uint8 type
cv.imshow('Original',gray)

# 스무딩 효과를 확인하기 위해 텍스트 삽입
cv.putText(gray,'soccer',(10,20),
           cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2) 

# 스무딩 적용
# parameter : 스무딩 적용 영상, 필터 크기, 가우시안 식의 표준편차(0.0: 자동 추정)
smooth=np.hstack((cv.GaussianBlur(gray,(5,5),0.0), 
                  cv.GaussianBlur(gray,(9,9),0.0),
                  cv.GaussianBlur(gray,(15,15),0.0)))
cv.imshow('Smooth',smooth)

# 엠보싱 적용 
femboss=np.array([[-1.0, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0],
                  [ 0.0, 0.0, 1.0]])

# filter2D 함수 : 주어진 영상의 배열과 같은 형의 배열 출력
# -> np.uint8 type은 음수가 발생하면 이상한 값으로 변환 저장됨 
# 음수를 표현하기 위해 2byte 형으로 변환 
gray16=np.int16(gray) 
emboss=np.uint8(np.clip(cv.filter2D(gray16, -1, femboss)+128, 0 ,255)) # 데이터형 변환+계산 결과 범위 특정 구간으로 제한 
emboss_bad=np.uint8(cv.filter2D(gray16, -1, femboss)+128) # 0-255 이내가 아닐 경우 오버플로우/언더플로우
emboss_worse=cv.filter2D(gray, -1, femboss)

cv.imshow('Emboss',emboss)
cv.imshow('Emboss_bad',emboss_bad)
cv.imshow('Emboss_worse',emboss_worse)

cv.waitKey()
cv.destroyAllWindows()