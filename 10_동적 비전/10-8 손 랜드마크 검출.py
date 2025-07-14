import cv2 as cv
import mediapipe as mp

mp_hand=mp.solutions.hands # 손 검출 모듈
mp_drawing=mp.solutions.drawing_utils
mp_styles=mp.solutions.drawing_styles

hand=mp_hand.Hands(max_num_hands=2, ## 손 2개까지 처리
                   static_image_mode=False, # 입력이 이미지인지 비디오인지 지정 : 비디오는 첫 프레임에 검출, 이후 추적 사용 
                   # True : 매 프레임마다 검출 진행 (정확도 향상)
                   min_detection_confidence=0.5, #검출 신뢰도 0.5이상일때 성공 간주
                   min_tracking_confidence=0.5) # 랜드마크 추적 신뢰도가 0.5 이하면 실패로 간주 -> 검출 다시 수행 

cap=cv.VideoCapture(0,cv.CAP_DSHOW)

while True:
    ret,frame=cap.read()
    if not ret:
      print('프레임 획득에 실패하여 루프를 나갑니다.')
      break
    
    res=hand.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB)) # 손 랜드마크 검출, 결과 저장 
    
    if res.multi_hand_landmarks:
        for landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,
                                      landmarks,
                                      mp_hand.HAND_CONNECTIONS,
                                      mp_styles.get_default_hand_landmarks_style(),
                                      mp_styles.get_default_hand_connections_style())

    cv.imshow('MediaPipe Hands',cv.flip(frame,1))	# 좌우반전
    if cv.waitKey(5)==ord('q'):
      break

cap.release()
cv.destroyAllWindows()