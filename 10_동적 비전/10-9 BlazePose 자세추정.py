import cv2 as cv
import mediapipe as mp

mp_pose=mp.solutions.pose # 자세 추정 모듈 
mp_drawing=mp.solutions.drawing_utils
mp_styles=mp.solutions.drawing_styles

pose=mp_pose.Pose(static_image_mode=False, # 입력 비디오로 설정
                  enable_segmentation=True, # 전경과 배경 분할
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5)

cap=cv.VideoCapture(0,cv.CAP_DSHOW)

while True:
    ret,frame=cap.read()
    if not ret:
      print('프레임 획득에 실패하여 루프를 나갑니다.')
      break
    
    res=pose.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
    
    mp_drawing.draw_landmarks(frame,
                              res.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
    
    cv.imshow('MediaPipe pose',cv.flip(frame,1)) # 좌우반전
    if cv.waitKey(5)==ord('q'):
      mp_drawing.plot_landmarks(res.pose_world_landmarks,mp_pose.POSE_CONNECTIONS)
      break

cap.release()
cv.destroyAllWindows()