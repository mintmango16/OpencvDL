from pixellib.semantic import semantic_segmentation # 의미 분할에 쓸 클래스 
import cv2 as cv
import os
os.chdir("09_인식")

seg=semantic_segmentation() # seg 객체 생성 
seg.load_ade20k_model('deeplabv3_xception65_ade20k.h5') #Ade20K 데이터셋으로 학습한 모델 로드

img_fname='busy_street.jpg'

#의미분할 수행 
seg.segmentAsAde20k(img_fname, # 원본 영상 
                    output_image_name='image_new.jpg') # 의미분할한 영상 저장 파일 지정 
info1,img_segmented1=seg.segmentAsAde20k(img_fname) # 분할 결과 저장 : 메타 정보 → info1, 분할된 영상 → img_segmented1
info2,img_segmented2=seg.segmentAsAde20k(img_fname,
                                         overlay=True) # 원래 영상에 분할 결과 투명하게 레이어 올리기  

# 결과 디스플레이 
cv.imshow('Image original',cv.imread(img_fname))
cv.imshow('Image segmention',img_segmented1)
cv.imshow('Image segmention overlayed',img_segmented2)

cv.waitKey()
cv.destroyAllWindows()