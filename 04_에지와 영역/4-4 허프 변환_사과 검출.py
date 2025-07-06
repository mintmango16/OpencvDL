import cv2 as cv 

import os
os.getcwd()
os.chdir("04_에지와 영역")

img=cv.imread('apples.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# 영상에서 원을 검출해 중심과 반지름 저장한 리스트 반환 
apples=cv.HoughCircles(gray,
                       cv.HOUGH_GRADIENT, # 에지 방향 정보 추가 사용
                       1, #누적 배열의 크기 지정 : 입력 영상과 같은 크기 설정
                       200, # 원 사이의 최소 거리
                       param1=150, #케니 에지 알고리즘의 T_high
                       param2=25, # 비최대 억제 임계값
                       minRadius=50,
                       maxRadius=115)

# 검출한 원 이미지에 그리기 
for i in apples[0]: 
    cv.circle(img,(int(i[0]),int(i[1])),int(i[2]),(255,0,0),2)

cv.imshow('Apple detection',img)  

cv.waitKey()
cv.destroyAllWindows()