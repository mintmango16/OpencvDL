import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
os.getcwd()
os.chdir("03_영상 처리")

img=cv.imread('JohnHancocksSignature.png',cv.IMREAD_UNCHANGED) 

t, bin_img=cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.imshow(bin_img, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show() 

b=bin_img[bin_img.shape[0]//2 : bin_img.shape[0],  # 
          0 : bin_img.shape[0]//2 + 1] # 영상의 일부만 따로 저장 : 좌측 하단 
plt.imshow(b,cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

se=np.uint8([[0,0,1,0,0],			# 구조 요소
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]])

b_dilation=cv.dilate(b,se,iterations=1)	# 팽창 # iterations: 적용 회수 
plt.imshow(b_dilation,cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b_erosion=cv.erode(b,se,iterations=1)	# 침식
plt.imshow(b_erosion,cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b_closing=cv.erode(b_dilation,se,iterations=1)	# 닫기
plt.imshow(b_closing,cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()