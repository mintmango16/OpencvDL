import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2 as cv
import os
os.chdir("09_인식")

seg = instanceSegmentation()

seg.load_model("pointrend_resnet50.pkl") 
cap=cv.VideoCapture(0)

# seg_video=instance_segmentation()
# seg_video.load_model("mask_rcnn_coco.h5")

target_class=seg.select_target_classes(person=True,book=True)
seg.process_camera(cap,segment_target_classes=target_class,frames_per_second=2,show_frames=True,frame_name='Pixellib',show_bboxes=True)

cap.release()
cv.destroyAllWindows()