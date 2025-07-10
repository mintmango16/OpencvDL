import cv2 as cv 
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
import os
os.chdir("08_컨볼루션 신경망")
model=ResNet50(weights='imagenet') # ResNet50을 백본으로 사용 

img=cv.imread('rabbit.jpg') 
x=np.reshape(cv.resize(img,(224,224)),(1,224,224,3)) # 신경망 입력 가능 형태로 변환 
x=preprocess_input(x) # ResNet50이 영상을 신경망에 입력하게 전에 수행하는 전처리 적용 

preds=model.predict(x) # 예측 : 1* 1000 배열 -> ImageNet 이 1000부류이기 때문에 해당 부류일 확률을 저장 
top5=decode_predictions(preds,top=5)[0] # 1000확률 중 가장 큰 5개 확률 취하고 해당하는 부류 이름 저장
print('예측 결과:',top5)

for i in range(5): # 영상에 인식 정보 작성 
    cv.putText(img,top5[i][1]+':'+str(top5[i][2]),(10,20+i*20),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

#영상 결과 디스플레이
cv.imshow('Recognition result',img)

cv.waitKey()
cv.destroyAllWindows()