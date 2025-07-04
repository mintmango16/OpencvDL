# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 22:19:14 2025

@author: asiae
"""

import cv2 as cv
import sys
import numpy as np

cap = cv.VideoCapture(0, cv.CAP_DSHOW) # 카메라와 연결 시도

if not cap.isOpened() :
    sys.exit('카메라 연결 실패')


frames =[]

while True :
    ret, frame = cap.read() # 비디오를 구성하는 프레임 획득 
    #ret (또는 retval): 불리언(boolean) 값 / True: 프레임을 성공적으로 읽어왔을 경우 반환
    #frame: 성공적으로 프레임을 읽어왔을 경우: 읽어온 **비디오 프레임(한 장의 이미지)**이 넘파이 배열 형태로 반환
        # 이미지의 픽셀 데이터를 담고 있으며, 일반적으로 (높이, 너비, 채널)의 형태 (예: BGR 채널의 컬러 이미지)

    if not ret :
        print('프레임 획득 실패, 루프 종료')
        break
    
    cv.imshow('video display', frame)
    
    key = cv.waitKey(1) # 1 ms동안 키보드 입력 기다림
    if key==ord('c'):
        frames.append(frame) # 프레임을 리스트에 추가 
    elif key==ord('q'): # 루프 종료 
        break
    
cap.release() # 카메라 연결 해제


if len(frames) > 0 : # 수집된 영상이 있으면 
    imgs = frames[0]
    for i in range(1, min(3, len(frames))): # 최대 3개를 이어붙임 
        imgs = np.hstack((imgs, frames[i]))
        
    cv.imshow('collected images', imgs)
    
    cv.waitKey()
    cv.destroyAllWindows()
    
    
# len(frames)
# Out[6]: 3

# frames[0].shape  : 하나의 이미지 크기 
# Out[7]: (480, 640, 3)

# type(imgs)
# Out[8]: numpy.ndarray

# imgs.shape
# Out[9]: (480, 1920, 3)
