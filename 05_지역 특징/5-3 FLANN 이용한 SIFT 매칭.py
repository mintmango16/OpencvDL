import cv2 as cv
import numpy as np
import time
import os
os.getcwd()
os.chdir("05_지역 특징")

img1=cv.imread('mot_color70.jpg')[190:350,440:560] # 버스를 크롭하여 모델 영상으로 사용
gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2=cv.imread('mot_color83.jpg')			     # 장면 영상
gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift=cv.SIFT_create()
kp1,des1=sift.detectAndCompute(gray1,None) # 두 영상 각각 SIFT 특징점 검출 + 기술자 추출 
kp2,des2=sift.detectAndCompute(gray2,None)
print('특징점 개수:',len(kp1),len(kp2)) 

start=time.time()
flann_matcher=cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED) # FLANN 객체 생성 
knn_match=flann_matcher.knnMatch(des1,des2,2) # 매칭 수행 : 최근접 2개 설정 

# 최근접 이웃 거리 비율 전략 사용하여 매칭 전략 설정 
T=0.7 # 임계값 
good_match=[]
for nearest1,nearest2 in knn_match:
    if (nearest1.distance/nearest2.distance)<T:
        good_match.append(nearest1)
print('매칭에 걸린 시간:',time.time()-start) 

# 매칭 결과를 보여줄 영상 생성 
img_match=np.empty((max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3),dtype=np.uint8) # 두 영상을 나란히 배치하는데 쓸 배열 생성
cv.drawMatches(img1,kp1,img2,kp2,good_match,img_match,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) # 두 영상에 특징점 표시, 매칭된 쌍을 선으로 연결하여 표시 

cv.imshow('Good Matches', img_match)

k=cv.waitKey()
cv.destroyAllWindows()

#특징점 개수: 232(모델영상) 4098(장면영상)
# 매칭에 걸린 시간: 0.017438173294067383 -> FLANN의 빠른 속도 확인 