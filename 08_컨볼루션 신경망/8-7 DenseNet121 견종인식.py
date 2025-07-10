from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121 # DenseNet121 모델 임포트 
from tensorflow.keras.utils import image_dataset_from_directory # 폴더에서 영상 읽기 
import pathlib # 폴더 다루는 모듈
import os 
import tensorflow as tf
 # DenseNet121 모델과 데이터셋의 크기가 커서 폴더에서 영상을 읽어 메인 메모리에 적재해 사용하는 방식 이용 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
data_path=pathlib.Path('datasets/stanford_dogs/images')  # Stanford dogs 데이터셋의 폴더 위치 지정 

train_ds=image_dataset_from_directory(data_path, #데이터 저장 경로 지정
                                      validation_split=0.2, # 훈련/테스트 데이터 분할 비율
                                      subset='training', # 훈련집합을 의미
                                      seed=123,# 난수 지정
                                      image_size=(224,224), # 영상을 변환할 크기 지정
                                      batch_size=16) # 미니 배치 크기 
test_ds=image_dataset_from_directory(data_path,
                                     validation_split=0.2,
                                     subset='validation', # 분할된 데이터 셋에서 검증 집합 
                                     seed=123,
                                     image_size=(224,224),
                                     batch_size=16)
# 신경망 모델 구성 
base_model=DenseNet121(weights='imagenet', # imagenet으로 사전 학습된 가중치 가져오도록 설정
                       include_top=False, # 모델 뒤쪽의 완전연결층 포함 여부
                       input_shape=(224,224,3)) # 입력층 
cnn=Sequential()
cnn.add(Rescaling(1.0/255.0)) 
cnn.add(base_model)
cnn.add(Flatten()) # 1차원으로 펼치기
cnn.add(Dense(1024,activation='relu'))
cnn.add(Dropout(0.75)) # 드롭아웃층 
cnn.add(Dense(units=120,activation='softmax')) # 부류가 120개 -> unit 120 설정 

cnn.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(learning_rate=0.000001),metrics=['accuracy'])
hist=cnn.fit(train_ds,epochs=200,validation_data=test_ds,verbose=2)

os.chdir("08_")
print('정확률=',cnn.evaluate(test_ds,verbose=0)[1]*100)

cnn.save('cnn_for_stanford_dogs.h5')	# 미세 조정된 모델을 파일에 저장

import pickle
f=open('dog_species_names.txt','wb') 
pickle.dump(train_ds.class_names,f) # 개의 품종 이름 저장 
f.close()

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()