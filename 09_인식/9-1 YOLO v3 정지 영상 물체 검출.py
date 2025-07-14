import numpy as np
import cv2 as cv
import sys
import os
os.chdir("09_인식")

def construct_yolo_v3(): # YOLO 모델 구성 
    f=open('coco_names.txt', 'r') # 부류 이름 불러오기
    class_names=[line.strip() for line in f.readlines()] 
    # 모델 정보 읽어 모델 생성 
    model=cv.dnn.readNet('yolov3.weights', # 신경망 가중치 
                         'yolov3.cfg')  # 신경망 구조 정보 
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()] # yolo_82, 94, 106층 저장 
    
    return model,out_layers,class_names #모델, 층, 부류 반환 

def yolo_detect(img,yolo_model,out_layers): # YOLO 모델로 img 영상에서 물체 검출 
    height,width=img.shape[0],img.shape[1] # 원본 영상의 높이와 너비 정보 저장 
    test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True) # yolo 모델에 입력 가능한 형태로 변환 : [0,255]→[0,1], 영상크기 448*448, BGR→RGB
    
    yolo_model.setInput(test_img) # 신경망에 변환 영상 입력
    output3=yolo_model.forward(out_layers) # 신경망의 전방 계산 수행 -> yolo_82, 94, 106층의 구조를 가지게 됨
    
    box,conf,id=[],[],[]		# 박스, 신뢰도, 부류 번호
    for output in output3:
        for vec85 in output: #85차원 벡터 반복 처리 (박스 4개, 신뢰도, 80개의 부류 확률)
            scores=vec85[5:]
            class_id=np.argmax(scores) # 80개 부류 확률 중 최댓값의 부류 번호저장 
            confidence=scores[class_id] # 최댓값의 해당 확률 
            if confidence>0.5:	# 신뢰도가 50% 이상인 경우만 취함
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
                w,h=int(vec85[2]*width),int(vec85[3]*height)
                x,y=int(centerx-w/2),int(centery-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
            
    ind=cv.dnn.NMSBoxes(box,conf,0.5,0.4) # 박스를 대상으로 비최대 억제를 적용해 중복성 제거 
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects

model,out_layers,class_names=construct_yolo_v3()		# YOLO 모델 생성
colors=np.random.uniform(0,255,size=(len(class_names),3))	# 부류마다 색깔

img=cv.imread('soccer.jpg')
if img is None: sys.exit('파일이 없습니다.')

res=yolo_detect(img,model,out_layers)	# YOLO 모델로 물체 검출

for i in range(len(res)):			# 검출된 물체를 영상에 표시
    x1,y1,x2,y2,confidence,id=res[i]
    text=str(class_names[id])+'%.3f'%confidence
    cv.rectangle(img,(x1,y1),(x2,y2),colors[id],2)
    cv.putText(img,text,(x1,y1+30),cv.FONT_HERSHEY_PLAIN,1.5,colors[id],2)

cv.imshow("Object detection by YOLO v.3",img)

cv.waitKey()
cv.destroyAllWindows()