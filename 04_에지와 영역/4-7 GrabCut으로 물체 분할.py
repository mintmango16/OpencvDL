import cv2 as cv 
import numpy as np
import os
os.getcwd()
os.chdir("04_에지와 영역")

img=cv.imread('soccer.jpg')	# 영상 읽기
img_show=np.copy(img)		# 붓 칠을 디스플레이할 목적의 영상

mask=np.zeros((img.shape[0],img.shape[1]),np.uint8) 
mask[:,:]=cv.GC_PR_BGD		# 모든 화소를 배경일 것 같음으로 초기화

BrushSiz=9				# 붓의 크기
LColor,RColor=(255,0,0), (0,0,255)	# 파란색(물체)과 빨간색(배경)
# 해당 화소들을 가지고 물체 히스토그램/배경 히스토그램 생성 
# -> 나머지 화소들은 두 히스토그램과의 유사성을 계산해 물체일 확률/배경일 확률 계산 
# -> 물체 영역과 배경 영역 갱신
# 이 과정을 반복, 영역이 거의 변하지 않으면 수렴한것으로 간주하고 종료  

def painting(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:  
        cv.circle(img_show,(x,y),BrushSiz,LColor,-1)	# 왼쪽 버튼 클릭하면 파란색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_FGD,-1)
    elif event==cv.EVENT_RBUTTONDOWN: 
        cv.circle(img_show,(x,y),BrushSiz,RColor,-1)	# 오른쪽 버튼 클릭하면 빨간색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_BGD,-1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_show,(x,y),BrushSiz,LColor,-1)    # 왼쪽 버튼 클릭하고 이동하면 파란색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_FGD,-1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
        cv.circle(img_show,(x,y),BrushSiz,RColor,-1)	# 오른쪽 버튼 클릭하고 이동하면 빨간색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_BGD,-1)

    cv.imshow('Painting',img_show)
    
cv.namedWindow('Painting')
cv.setMouseCallback('Painting',painting)

while(True):				# 붓 칠을 끝내려면 'q' 키를 누름
    if cv.waitKey(1)==ord('q'): 
        break

# GrabCut 적용하는 코드
background=np.zeros((1,65),np.float64)	# 배경 히스토그램 0으로 초기화 : bin 65로 설정 
foreground=np.zeros((1,65),np.float64)	# 물체 히스토그램 0으로 초기화

# 실제 분할 수행
cv.grabCut(img, # 원본 영상
           mask, # 사용자가 지정한 물체와 배경 정보 지정 맵
           None, # 관심영역 지정 
           background, # 배경 히스토그램
           foreground, # 물체 히스토그램
           5, # 반복 횟수 
           cv.GC_INIT_WITH_MASK) # 배경과 물체를 표시한 맵을 사용
mask2=np.where((mask==cv.GC_BGD)|(mask==cv.GC_PR_BGD), 
               0,# 배경 아니면 배경으로 추측되는 화소는 0으로 저장, 물체/물체로 추츨되는 화소는 1로 변환하여 저장 
               1).astype('uint8')
grab=img*mask2[:,:,np.newaxis] # 배경 화소를 검개 바꾸어 저장 
cv.imshow('Grab cut image',grab)  

cv.waitKey()
cv.destroyAllWindows()