import cv2 as cv
import numpy as np
import os
os.getcwd()
os.chdir("04_에지와 영역 ")

img=cv.imread('soccer.jpg')	 # 영상 읽기
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
canny=cv.Canny(gray,100,200) # 캐니 알고리즘으로 엣지 계산 

# 경계선을 찾아 contour에 저장 
contour,hierarchy=cv.findContours(canny, # 경계선 찾을 엣지 영상
                                  cv.RETR_LIST, # 구멍 경계선 찾는 방식 지정 : 맨 바깥쪽 경계선만 
                                  cv.CHAIN_APPROX_NONE) # 경계선 표현 방식 지정 

lcontour=[]   
for i in range(len(contour)):
    if contour[i].shape[0]>100:	# 길이가 100보다 크면 = 실제 길이 50 이상 
        lcontour.append(contour[i])

# 영상에 경계선 그리기 
cv.drawContours(img, lcontour,
                -1, # 모든 경계선
                (0,255,0), # 색
                2)  # 두께
             
cv.imshow('Original with contours',img)    
cv.imshow('Canny',canny)    

cv.waitKey()
cv.destroyAllWindows()