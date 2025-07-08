import cv2 as cv
import os
os.getcwd()
os.chdir("05_지역 특징")

img=cv.imread('mot_color70.jpg') # 영상 읽기
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)  # 흑백으로 변환

# SIFT 특징점을 추출하는데 쓸 객체 생성 : 모든 parameter가 기본값 존재 
# nfeatures : 검출할 특징점 개수 지정 (0:모두 반환), nOctaveLayers : 옥타브 개수 지정, 
# comtrastThreshold : 테일러 확장으로 미세 조정 (클수록 적은 수의 특징점), edgeThreshold : 엣지에서 검출된 특징점 걸러내기, sigma : 옥타브 0의 입력 영상에 적용할 가우시안의 표준편차 
sift=cv.SIFT_create() 
kp,des=sift.detectAndCompute(gray,None) # 특징점과 기술자 각각 저장 

# kp = sift.detect(gray, None)
# des = sift.compute(gray, kp)

gray=cv.drawKeypoints(gray,kp,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 검출한 특징점 영상에 표시 
cv.imshow('sift', gray)

k=cv.waitKey()
cv.destroyAllWindows()

#원의 중심 : 특징점 위치, 반지름 : 스케일, 원안의 선분 : 지배적인 방향