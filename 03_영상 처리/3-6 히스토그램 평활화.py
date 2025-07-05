import cv2 as cv
import matplotlib.pyplot as plt

import os
os.getcwd()
os.chdir("03_영상 처리")

img=cv.imread('mistyroad.jpg') # 안개 낀 도로 이미지 

plt.figure(figsize=(12, 8))

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)		# 명암 영상으로 변환
plt.subplot(2, 2, 1), plt.imshow(gray,cmap='gray'), plt.xticks([]), plt.yticks([])

h=cv.calcHist([gray],[0],None,[256],[0,256])	# 히스토그램을 구해 출력
plt.subplot(2, 2, 2), plt.plot(h,color='r',linewidth=1)
# 대부분의 화솟값이 100-200에 분포해 있음을 확인

equal=cv.equalizeHist(gray)			# 히스토그램을 평활화하고 출력
plt.subplot(2, 2, 3), plt.imshow(equal,cmap='gray'), plt.xticks([]), plt.yticks([])

h=cv.calcHist([equal],[0],None,[256],[0,256])	# 히스토그램을 구해 출력
plt.subplot(2, 2, 4), plt.plot(h,color='r',linewidth=1)

plt.tight_layout() # 서브플롯들이 겹치지 않도록 자동 조정
plt.show()
