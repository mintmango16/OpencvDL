# 최적화 분할 : 지역적 명암 변화 + 전역적 정보 -> 영상을 그래프로 구현/분할을 최적화 문제로 풀이

# 영상 그래프 표현 : 화소/슈퍼화소를 노드로 취함 
# 두 노드를 연결하는 엣지의 가중치로는 유사도 사용

# 정규화 절단 알고리즘
# 화소를 노드로 사용, 색상+위치 결합한 5차원 벡터, 유사도를 엣지 가중치로 사용
# cut 식 : 원래 영역은 2개로 나누었을 떄 영역 분할의 적절한 정도를 측정해주는 목적 함수 
# -> 정규화 ncut : 영역 크기에 중립이 되도록 설정 -> ncut이 작을 수록 좋은 분할(최적화문제)

import skimage
import numpy as np
import cv2 as cv
import time

coffee=skimage.data.coffee()

# 분할 시 소요되는 시간 계산 
start=time.time()
slic=skimage.segmentation.slic(coffee,
                               compactness=20, 
                               n_segments=600, # 슈퍼화소 개수 
                               start_label=1)
g=skimage.graph.rag_mean_color(coffee,
                               slic, # 슈퍼화소를 노드로 사용 
                               mode='similarity') #유사도를 엣지 가중치로 사용한 그래프 
ncut=skimage.graph.cut_normalized(slic,g)	# 정규화 절단, 화소에 영역의 번호 부여 
print(coffee.shape,' Coffee 영상을 분할하는데 ',time.time()-start,'초 소요')

marking=skimage.segmentation.mark_boundaries(coffee,ncut) # 영역 경계 표시 
ncut_coffee=np.uint8(marking*255.0)

print(np.unique(ncut))
print(np.unique(ncut).size) # 영역의 개수 

cv.imshow('Normalized cut',cv.cvtColor(ncut_coffee,cv.COLOR_RGB2BGR))  

cv.waitKey()
cv.destroyAllWindows()

#영역 분할이 영상의 색상 정보에만 의존하는 단점 존재 