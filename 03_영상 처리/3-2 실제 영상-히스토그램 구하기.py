# 이진 영상 : 화소가 0=흑 또는 1=백으로 이루어진 영상(화소당 1비트 저장)
# 컬러 영상이나 명암 영상을 이진으로 변환할 필요가 있음 
# 명암 영상을 이진화하려면 임계값보다 클경우 1, 작을경우 0으로 바꾸는 과정을 거침-> 임계값 결정이 중요한 키포인트 
# 임계값을 잘못 설정하면 화소가 물체/배경 한쪽에 쏠리는 현상 발생
# 그런 현상을 확인하기 위해 히스토그램 이용 

import cv2 as cv
import matplotlib.pyplot as plt
import os
os.getcwd()
os.chdir("03_영상 처리")

img=cv.imread('soccer.jpg') 

# 이미지의 히스토그램 계산 
#cv.calcHist(images, channels, mask, histSize, ranges)
h=cv.calcHist([img], # 분석할 대상 이미지 -> 리스트 형태로 받음 
              [2], # 계산할 채널 -> red
              None, # 특정 영역 설정
              [256], # 히스토그램의 bin
              [0,256]) # 픽셀 값 범위 설정 
# 결과적으로 각 인덱스에 채널값의 픽셀수가 저장됨
plt.plot(h,color='r',linewidth=1)
plt.show()