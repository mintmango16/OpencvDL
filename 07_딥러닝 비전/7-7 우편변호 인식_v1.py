import numpy as np
import tensorflow as tf
import cv2 as cv 
import matplotlib.pyplot as plt
import winsound

model=tf.keras.models.load_model('dmlp_trained.h5') # 7-5에서 저장된 필기체 인식 모델 임포트 : 98.42 정확률 

def reset(): # 초기화 함수 -> 박스 지우기 
    global img 
       
    img=np.ones((200,520,3),dtype=np.uint8)*255 # 200*520 의 3채널 컬러영상 저장할 배열*255 -> 모든화소가 흰색인 배열 
    for i in range(5):
        cv.rectangle(img,(10+i*100,50),(10+(i+1)*100,150),(0,0,255)) # 지정 위치에 5개의 빨간색 박스 그리기 
    cv.putText(img,'e:erase s:show r:recognition q:quit',(10,40),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1) # 안내 문구 

def grab_numerals():# 5개 숫자를 떼어냄 
    numerals=[]
    for i in range(5): # 각 박스에서 숫자를 떼냄 
        roi=img[51:149, 11+i*100:9+(i+1)*100, 0] #  ROI(Region Of Interest)를 잘라내기 위한 슬라이싱 
        roi=255-cv.resize(roi,(28,28), interpolation=cv.INTER_CUBIC) # 색상 반전 수행
        numerals.append(roi)  
    numerals=np.array(numerals) # 각 박스에 대한 숫자 28*28 크기 정보 저장 
    return numerals

def show(): # 박스에서 숫자를 명암영상으로 표시 
    numerals=grab_numerals() 
    plt.figure(figsize=(25,5))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(numerals[i],cmap='gray')
        plt.xticks([]); plt.yticks([])
    plt.show()
    
def recognition(): # 모델이 인식 하여 예측 
    numerals=grab_numerals()
    numerals=numerals.reshape(5,784) # 2차원 -> 1차원으로 펼치기
    numerals=numerals.astype(np.float32)/255.0 # 0,1 범위로 변환(정규화)
    res=model.predict(numerals) # 신경망 모델로 예측 후 결과 저장 : 5*10배열
    class_id=np.argmax(res,axis=1) # 예측값의 최대값의 인덱스를 찾아 저장 
    for i in range(5): # 인식 결과(예측 결과) 화면에 표시 
        cv.putText(img,str(class_id[i]),(50+i*100,180),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    winsound.Beep(1000,500)    
        
BrushSiz=4 # 원의 크기 
LColor=(0,0,0) # 원의 색상 

def writing(event,x,y,flags,param): # 마우스 콜백 함수 
    if event==cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),BrushSiz,LColor,-1) 
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON: 
        cv.circle(img,(x,y),BrushSiz,LColor,-1)

reset()
cv.namedWindow('Writing') # 윈도우 생성
cv.setMouseCallback('Writing',writing) #마우스 입력이 들어오면 함수 실행 

while(True):
    cv.imshow('Writing',img)
    key=cv.waitKey(1)
    if key==ord('e'):
        reset()
    elif key==ord('s'):
        show()        
    elif key==ord('r'):
        recognition()
    elif key==ord('q'):
        break
    
cv.destroyAllWindows()