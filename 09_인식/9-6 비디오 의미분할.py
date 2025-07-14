# 의미 분할 프로그래밍 : 비디오 

from pixellib.semantic import semantic_segmentation
import cv2 as cv
import os
os.chdir("09_인식")

cap=cv.VideoCapture(0) # 캡과 연결하여 결과를 cap 객체에 저장 

seg_video=semantic_segmentation()# seg 객체 생성
seg_video.load_ade20k_model('deeplabv3_xception65_ade20k.h5')#Ade20K 데이터셋으로 학습한 모델 로드

seg_video.process_camera_ade20k(cap, # 캠 비디오 
                                overlay=True, # 원래 영상에 반투명하게 오버레이
                                frames_per_second=2, # 초당 2프레임 저장 
                                output_video_name='output_video.mp4', # 폴더에 저장할 mp4 파일 이름 지정
                                show_frames=True, # 윈도우 창으로 분할 결과 실시간으로 디스플레이
                                frame_name='Pixellib') # 윈도우 창 이름 
#  process_camera_ade20k 함수의 경우 q키를 누르면 마치도록 자동 설정됨 

cap.release()
cv.destroyAllWindows()