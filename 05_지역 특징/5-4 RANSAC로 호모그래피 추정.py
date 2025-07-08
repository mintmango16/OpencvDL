import cv2 as cv
import numpy as np
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

flann_matcher=cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED) # FLANN 객체 생성 
knn_match=flann_matcher.knnMatch(des1,des2,2) # 매칭 수행 : 최근접 2개 설정 

# 최근접 이웃 거리 비율 전략 사용하여 매칭 전략 설정 
T=0.7 # 임계값 
good_match=[]
for nearest1,nearest2 in knn_match:
    if (nearest1.distance/nearest2.distance)<T:
        good_match.append(nearest1)
        
# good_match : 매칭 쌍 중에 좋은 것을 골라 저장한 리스트임 
# distance(두 특징점 거리), queryIdx(모델 영상의 특징점 번호=ai의 i), trainIdx(장면 영상의 특징점 번호=bj의 j) 

points1=np.float32([kp1[gm.queryIdx].pt for gm in good_match]) # 모델 영상의 특징점 번호 저장 
points2=np.float32([kp2[gm.trainIdx].pt for gm in good_match]) # 장면 영상의 특징점 번호 저장 

H,_=cv.findHomography(points1,points2,cv.RANSAC)  # 호모그래피 행렬 추정하여 저장 (RANSAC 알고리즘 이용)

h1,w1=img1.shape[0],img1.shape[1] 		# 첫 번째 영상의 크기 저장
h2,w2=img2.shape[0],img2.shape[1] 		# 두 번째 영상의 크기

box1=np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(4,1,2)
box2=cv.perspectiveTransform(box1,H)

img2=cv.polylines(img2,[np.int32(box2)],True,(0,255,0),8) # box2 를 두번째 영상에 그림

img_match=np.empty((max(h1,h2),w1+w2,3),dtype=np.uint8) # 두 영상을 나란히 배치하는데 쓸 배열 생성
cv.drawMatches(img1,kp1,img2,kp2,good_match,img_match,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)# 두 영상에 특징점 표시, 매칭된 쌍을 선으로 연결하여 표시 
   
cv.imshow('Matches and Homography',img_match)

k=cv.waitKey()
cv.destroyAllWindows()

# 실행 결과
# 버스 모델 영상을 장면 영상에서 잘 검출하고 호모그래피 행렬도 잘 추정했음을 확인 가능
# 아웃라이어 (버스 안과 가로등 쪽이 연결된 매칭 쌍)도 잘 걸러냈음이 확인됨 
# 초록색 박스가 바로 호모그래피 행렬의 결과물임