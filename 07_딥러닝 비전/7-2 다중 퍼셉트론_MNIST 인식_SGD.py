# 다중 퍼셉트론으로 MNIST 인식하기 : SGD 옵티마이저 

import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

# Sequential 과 function API의 두 모델 제공 : 다층 퍼셉트론처럼 왼쪽에서 오른쪽으로 계산에 방향성이 있는 경우 Sequential 사용
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense # 여러가지 층 제공, 다층 퍼셉트론의 완전연결층은 Dense클래스로 쌓음
from tensorflow.keras.optimizers import SGD # 학습 알고리즘이 사용하는 옵티마이저 함수 제공 (SGD, Adam, AdaGrad, RMSprop emd)

# 데이터 준비 
(x_train,y_train),(x_test,y_test)=ds.mnist.load_data()
x_train=x_train.reshape(60000,784) # 28*28 -> 1차원 구조 변환
x_test=x_test.reshape(10000,784)
x_train=x_train.astype(np.float32)/255.0 # [0,1]로 정규화 + 실수 데이터형 변환
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10) # one-hot 코드 변환 : 2차원 구조의 맵을 1차원으로 펼침 
y_test=tf.keras.utils.to_categorical(y_test,10) 

# 모델 선택 및 구축 (신경망 구조 설계)
mlp=Sequential()
# 층 쌓기 : dense로 완전연결층 
mlp.add(Dense(units=512, # 은닉층 노드 배치 개수
              activation='tanh', # 은닉층 활성 함수
              input_shape=(784,))) # 입력층 노드 배치 개수
# 출력층에 해당하는 완전연결층 :input_shape 생략 가능
mlp.add(Dense(units=10, # 출력층 노드 배치
              activation='softmax')) # 출력층 활성 함수 : 보통 softmax 사용 

# 학습 초기 세팅 
mlp.compile(loss='MSE', # 손실함수 선정 
            optimizer=SGD(learning_rate=0.01),# 옵티마이저 SGD 설정 +학습률 0.01
            metrics=['accuracy'])  #성과 지표 설정 
# 학습 진행 
mlp.fit(x_train,y_train,
        batch_size=128,  # 미니 배치 크기
        epochs=50, # 세대 (반복) 수 
        validation_data=(x_test,y_test),
        verbose=2)

# 예측 
res=mlp.evaluate(x_test,y_test,verbose=0) 
print('정확률=',res[1]*100)

#Epoch 50/50
# 469/469 - 1s - 3ms/step - accuracy: 0.8871 - loss: 0.0189 - val_accuracy: 0.8938 - val_loss: 0.0178
# 정확률= 89.38000202178955