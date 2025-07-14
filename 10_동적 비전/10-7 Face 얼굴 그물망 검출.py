import cv2 as cv
import mediapipe as mp

mp_mesh=mp.solutions.face_mesh # 얼굴 그물망 검출 프로그램 
mp_drawing=mp.solutions.drawing_utils
mp_styles=mp.solutions.drawing_styles

mesh=mp_mesh.FaceMesh(max_num_faces=2, # 얼굴 2개까지 처리
                      refine_landmarks=True, # 눈, 입의 랜드마크를 정교하게 검출 
                      min_detection_confidence=0.5, # 얼굴 검출 신뢰도 0.5이상일때 성공 간주
                      min_tracking_confidence=0.5) # 랜드마크 추적 신뢰도가 0.5 이하면 실패로 간주 -> 검출 다시 수행 

cap=cv.VideoCapture(0,cv.CAP_DSHOW)

while True:
    ret,frame=cap.read()
    if not ret:
      print('프레임 획득에 실패하여 루프를 나갑니다.')
      break
    
    res=mesh.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB)) # 그물망 검출 후 결과 저장 
    
    if res.multi_face_landmarks:
        for landmarks in res.multi_face_landmarks: # 검출한 그물망 그리기 
            mp_drawing.draw_landmarks(image=frame,
                                      landmark_list=landmarks,
                                      connections=mp_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()) # 그물망 그리기 
            mp_drawing.draw_landmarks(image=frame,
                                      landmark_list=landmarks,
                                      connections=mp_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()) # 얼굴 경계+눈+눈썹 그리기 
            mp_drawing.draw_landmarks(image=frame,
                                      landmark_list=landmarks,
                                      connections=mp_mesh.FACEMESH_IRISES,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style()) #눈동자만 그리기 
        
    cv.imshow('MediaPipe Face Mesh',cv.flip(frame,1))		# 좌우반전
    if cv.waitKey(5)==ord('q'):
      break

cap.release()
cv.destroyAllWindows()