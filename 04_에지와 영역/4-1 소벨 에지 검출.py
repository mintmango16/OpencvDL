# 엣지 검출 기법 : 소벨 연산자(1차 기반 미분 엣지 연산자) 적용하여 엣지 강도 맵 구하기 
import cv2 as cv

import os
os.getcwd()
os.chdir("04_에지와 영역 ")

img=cv.imread('soccer.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 명암 영상으로 변환 

# cv.Sobel(src, ddepth, dx, dy, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
# 소벨 연산자 함수 parameter : 
# 입력 영상, 출력 영상 데이터 타입, x방향 미분 차수, y방향 미분 차수, 필터 크기
grad_x=cv.Sobel(gray,cv.CV_32F,1,0,ksize=3)	# 소벨 연산자 적용 : x방향 (수평 변화)
grad_y=cv.Sobel(gray,cv.CV_32F,0,1,ksize=3) # y방향(수직 변화)

sobel_x=cv.convertScaleAbs(grad_x)	# 절대값을 취해 양수 영상으로 변환 : 음수는 0, 255이상은 255로 변환
sobel_y=cv.convertScaleAbs(grad_y)

edge_strength=cv.addWeighted(sobel_x,0.5,sobel_y,0.5,0)	#sobel_x*0.5 + sobel_y*0.5 +0 반환 
# 에지 강도 계산, 둘의 데이터 형이 같아야 함 

cv.imshow('Original',gray)
cv.imshow('sobelx',sobel_x)
cv.imshow('sobely',sobel_y)
cv.imshow('edge strength',edge_strength)

cv.waitKey()
cv.destroyAllWindows()