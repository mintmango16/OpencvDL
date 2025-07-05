import cv2 as cv
import os
os.getcwd()
os.chdir("03_영상 처리")

img=cv.imread('rose.png')
patch=img[250:350,170:270,:] # 100x100단위로 잘라 가져옴

img=cv.rectangle(img,(170,250),(270,350),(255,0,0),3) # 오려낸 곳 표시 

# 모두 5배로 확대 
patch1=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_NEAREST) # 최근접 이웃
patch2=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_LINEAR) # 양선형 보간 방법
patch3=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_CUBIC) # 양3차 보간 방법 

#최근접 이웃의 경우 꽃잎 가장자리에 계단 모양의 에일리어싱 현상이 심함 
#양선형과 양3차의 화질은 육안으로 구별 어려움 

cv.imshow('Original',img)
cv.imshow('Resize nearest',patch1) 
cv.imshow('Resize bilinear',patch2) 
cv.imshow('Resize bicubic',patch3) 

cv.waitKey()
cv.destroyAllWindows()