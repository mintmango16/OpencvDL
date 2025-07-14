import cv2 as cv
import mediapipe as mp
import os
os.chdir('10_동적 비전')
img=cv.imread('BSDS_376001.jpg')
dice=cv.imread('dice.png',cv.IMREAD_UNCHANGED)	# 증강 현실에 쓸 장신구
dice=cv.resize(dice,dsize=(0,0),fx=0.1,fy=0.1) # 영상 10% 축소, 
w,h=dice.shape[1],dice.shape[0] # 너비와 높이 저장 

mp_face_detection=mp.solutions.face_detection 
mp_drawing=mp.solutions.drawing_utils

face_detection=mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5)

cap=cv.VideoCapture(0,cv.CAP_DSHOW)

while True:
    ret,frame=cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
    
    res=face_detection.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
    
    if res.detections:
        for det in res.detections: # 검출된 얼굴에 장신구 달고 디스플레이 
            p=mp_face_detection.get_key_point(det,mp_face_detection.FaceKeyPoint.RIGHT_EYE) # 얼굴 정보 det에서 오른쪽 눈 위치를 p에 저장
            x1,x2=int(p.x*frame.shape[1]-w//2),int(p.x*frame.shape[1]+w//2) # 오른쪽 눈을 중심으로 장신구를 배치하기 위한 좌표 계산
            y1,y2=int(p.y*frame.shape[0]-h//2),int(p.y*frame.shape[0]+h//2)
            if x1>0 and y1>0 and x2<frame.shape[1] and y2<frame.shape[0]: # 장신구 영상이 원본 영상에 존재하는지 확인
                alpha=dice[:,:,3:]/255 # 투명도를 나타내는 알파값
                frame[y1:y2,x1:x2]=frame[y1:y2,x1:x2]*(1-alpha) + dice[:,:,:3]*alpha # 원본영상 + 장신구영상 : 투명도 계산하여 합산 
            
    cv.imshow('MediaPipe Face AR',cv.flip(frame,1))
    if cv.waitKey(5)==ord('q'):
        break

cap.release()
cv.destroyAllWindows()