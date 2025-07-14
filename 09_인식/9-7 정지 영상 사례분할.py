# from pixellib.instance import instance_segmentation # 사례 분할 클래스 
# import cv2 as cv
# import os
# os.chdir("09_인식")

# seg=instance_segmentation()
# seg.load_model("mask_rcnn_coco.h5") # COCO 데이터셋으로 학습한 모델 로드 및 저장 

import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2 as cv
import os
os.chdir("09_인식")


seg = instanceSegmentation()
seg.load_model("pointrend_resnet50.pkl") 

img_fname='busy_street.jpg'

# 사례 분할 실행 후 결과 저장 
info, img_segmented=seg.segmentImage(img_fname,
                                     show_bboxes=True) # 각 영역에 박스 처리 + 물체 확률 표기 

cv.imshow('Image segmention overlayed',img_segmented)

cv.waitKey()
cv.destroyAllWindows()