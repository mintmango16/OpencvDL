import skimage
import numpy as np
import cv2 as cv

orig=skimage.data.horse()
img=255-np.uint8(orig)*255 # 말(0) 배경(1) 에서 말(255) 배경 (0)인 영상으로 변환 
cv.imshow('Horse',img)

contours,hierarchy=cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) # 물체의 경계선 추출 

img2=cv.cvtColor(img,cv.COLOR_GRAY2BGR)		# 컬러 디스플레이용 영상
cv.drawContours(img2,contours,-1,  # 경계선 표시 
                (255,0,255),2) # 색과 두께
cv.imshow('Horse with contour',img2)

contour=contours[0] # 경계선 하나만 저장 

m=cv.moments(contour)				# 몇 가지 특징 : 모멘트 추출 
area=cv.contourArea(contour)        # 경계선으로 둘러싸인 영역 면적 계산 
cx,cy=m['m10']/m['m00'],m['m01']/m['m00']
perimeter=cv.arcLength(contour,
                       True) # 폐곡선 유무 
roundness=(4.0*np.pi*area)/(perimeter*perimeter) # 둥근 정도 계산 
print('면적=',area,'\n중점=(',cx,',',cy,')','\n둘레=',perimeter,'\n둥근 정도=',roundness)

img3=cv.cvtColor(img,cv.COLOR_GRAY2BGR)		# 컬러 디스플레이용 영상

contour_approx=cv.approxPolyDP(contour,8,True)	# 직선 근사
cv.drawContours(img3,[contour_approx],-1,(0,255,0),2)

hull=cv.convexHull(contour)			# 볼록 헐
hull=hull.reshape(1,hull.shape[0],hull.shape[2])
cv.drawContours(img3,hull,-1,(0,0,255),2) # 빨간색 표시 

cv.imshow('Horse with line segments and convex hull',img3)

cv.waitKey()
cv.destroyAllWindows()