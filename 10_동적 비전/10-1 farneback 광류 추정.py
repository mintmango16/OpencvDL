import numpy as np
import cv2 as cv
import sys

def draw_OpticalFlow(img,flow,step=16): # 광류 맵을 원본 영상에 그리는 함수 
    #flow : 광류 맵, 원본 영상과 같은 크기, 화소마다 y와 x방향의 이동량=모션벡터
    for y in range(step//2,frame.shape[0],step): # step 만큼 건너뛰어 화소 접근 
        for x in range(step//2,frame.shape[1],step):
            dx,dy=flow[y,x].astype(int) # 해당 화소의 모션벡터 저장 
            if(dx*dx+dy*dy) > 1: #모션벡터가 1보다 클 경우 = 큰 모션 기준 지정 
                cv.line(img,(x,y),(x+dx,y+dy),(0,0,255),2) # 큰 모션 있는 곳은 빨간색
            else:
                cv.line(img,(x,y),(x+dx,y+dy),(0,255,0),2) # 작을 경우 초록색 표시          
    
cap=cv.VideoCapture(0,cv.CAP_DSHOW)	# 카메라와 연결 시도
if not cap.isOpened(): sys.exit('카메라 연결 실패')
    
prev=None

while(1):
    ret,frame=cap.read()	# 비디오를 구성하는 프레임 획득
    if not ret: sys('프레임 획득에 실패하여 루프를 나갑니다.')
    

    if prev is None:	# 첫 프레임이면 광류 계산 없이 prev만 설정
        prev=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        continue
    
    curr=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    flow=cv.calcOpticalFlowFarneback(prev, # 광류 계산할 이전 프레임
                                     curr, # 광류 계산할 현재 프레암
                                     None, # 출력 배열 방식 지정
                                     0.5, # 이미지 피라미드 구축시 다음 레이어의 스케일
                                     3, # 구축할 이미지 피라미드의 레벨(층) 수 
                                     15, # 윈도우(창)의 크기
                                     3, # 알고리즘이 수행하는 반복 횟수 
                                     5, # 이웃 픽셀 영역의 크기(5*5)
                                     1.2, # 가우시안 필터 표준편차
                                     0) # 알고리즘의 작동 플래그 : 기본 설정 사용 
    
    draw_OpticalFlow(frame,flow)
    cv.imshow('Optical flow',frame)

    prev=curr

    key=cv.waitKey(1)	# 1밀리초 동안 키보드 입력 기다림
    if key==ord('q'):	# 'q' 키가 들어오면 루프를 빠져나감
        break 
    
cap.release()			# 카메라와 연결을 끊음
cv.destroyAllWindows() 