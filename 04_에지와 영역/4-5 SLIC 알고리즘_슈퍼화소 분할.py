# SLIC 알고리즘으로 슈퍼화소 분할 

# 영상을 아주 작은 영역(슈퍼화소)로 분할해 다른 알고리즘의 입력으로 사용할 수 있음
# SLIC은 k-means와 비슷하게 작동 + 처리과정 단순, 고성능
# 입력영상에서 k개 화소를 중심으로 등간격으로 패치 분할 -> 다음 패치 중심을 군집 중심으로 간주
# SLIC의 경우 이미지의 모든 픽셀을 처리할 때 (R,G,B,x,y) 5차원 벡터로 표현함
# 클러스터의 중심은 화소들 중 선택되거나 할당된 화소의 평균으로 5차원 벡터로 표현됨
# 화소를 가장 가까운 군집 중심에 할당하는 단계+군집 중심을 갱신하는 단계 반복 
# 1) 화소 할당 단계 : 화소 각각에 대해 주위 4개 군집 중심과 자신의 색상 + 거리를 계산하여 유사한 군집중심에 할당
        # -> 색상 유사도와 공간적 유사도를 결합한 거리 측도 사용 
# 2) 군집 중심 갱신 단계 : 화소 할당이 끝나면 각 군집 중심은 자신에게 할당된 화소를 평균해 군집 중심 갱신

# 최종적으로 모든 군집 중심의 이동량의 평균이 임계치가 작으면 수렴했다고 판단하여 중지 
import skimage
import numpy as np
import cv2 as cv

import os
os.getcwd()
os.chdir("04_에지와 영역")

img=skimage.data.coffee() # 해당 라이브러리 내의 영상을 가져옴 :RGB 형식 (numpy 배열)
cv.imshow('Coffee image',cv.cvtColor(img,cv.COLOR_RGB2BGR))  

# 슈퍼 화소 분할 수행 
slic1=skimage.segmentation.slic(img,  # 분할 대상 영상 
                                compactness=20, # 슈퍼 화소의 모양 조절 (클수록 사각형+색상 균일성 희생)
                                n_segments=600) # 슈퍼 화소의 개수 지정 = K

# slic1의 분할 정보 img에 표시 
sp_img1=skimage.segmentation.mark_boundaries(img,slic1)

sp_img1=np.uint8(sp_img1*255.0) # 0-1 사이의 실수 -> 0-255 

# 슈퍼 화소의 모양 차이 확인 
slic2=skimage.segmentation.slic(img,compactness=40,n_segments=600)
sp_img2=skimage.segmentation.mark_boundaries(img,slic2)
sp_img2=np.uint8(sp_img2*255.0)

cv.imshow('Super pixels (compact 20)',cv.cvtColor(sp_img1,cv.COLOR_RGB2BGR))
cv.imshow('Super pixels (compact 40)',cv.cvtColor(sp_img2,cv.COLOR_RGB2BGR))

cv.waitKey()
cv.destroyAllWindows()