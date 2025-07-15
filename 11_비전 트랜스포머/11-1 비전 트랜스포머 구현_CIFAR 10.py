import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

(x_train,y_train),(x_test,y_test)=keras.datasets.cifar10.load_data()

n_class=10                      # 부류 수
img_siz=(32,32,3)               # 영상의 크기

# 그림 11-18(a)의 구현 (영상크기 확대)
patch_siz=4                     # 패치 크기
p2=(img_siz[0]//patch_siz)**2   # 패치 개수 = p^2, T에 해당 
d_model=64                      # 임베딩 벡터 차원
h=8                             # 헤드 개수 : 기존 텐서플로 제공 트랜스포머와 동일 
N=6                             # 인코더 블록의 개수

class Patches(layers.Layer): # 기존 영상을 p2개의 패치로 잘라 만든 patches 반환
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.p_siz=patch_size

    def call(self, img):
        batch_size=tf.shape(img)[0]
        patches=tf.image.extract_patches(images=img,sizes=[1,self.p_siz,self.p_siz,1],strides=[1,self.p_siz,self.p_siz,1],rates=[1,1,1,1],padding="VALID")
        patch_dims=patches.shape[-1]
        patches=tf.reshape(patches,[batch_size,-1,patch_dims])
        return patches

class PatchEncoder(layers.Layer): # 패치 화소를 1차원으로 이어붙이고 위치 인코딩 적용 
    def __init__(self,p2,d_model): # p2: 패치 개수
        super(PatchEncoder,self).__init__()
        self.p2=p2
        self.projection=layers.Dense(units=d_model) # 차원 변환 
        self.position_embedding=layers.Embedding(input_dim=p2,output_dim=d_model) # 위치 인코딩 벡터 생성

    def call(self,patch):
        positions=tf.range(start=0,limit=self.p2,delta=1)
        encoded=self.projection(patch) + self.position_embedding(positions) # 차원 + 위치 인코딩 벡터 
        return encoded

def create_vit_classifier(): # 모델 생성 
    input=layers.Input(shape=(img_siz)) # 영상 크기 지정
    nor=layers.Normalization()(input)  # (0,255) -> (0,1) 정규화
    
    patches=Patches(patch_siz)(nor)	# 패치 생성
    x=PatchEncoder(p2,d_model)(patches)	# 패치 인코딩 → 입력 영상을 트랜스포머의 입력 형태로 변환 

    for _ in range(N):			# 다중 인코더 블록
        x1=layers.LayerNormalization(epsilon=1e-6)(x)		# 층 정규화
        
        # 트랜스포머 구현 핵심 함수 
        x2=layers.MultiHeadAttention(num_heads=h, # 헤드 개수
                                     key_dim=d_model//h, # key의 차원 설정
                                     dropout=0.1)(x1,x1) # dropout 비율 / 입력 지정 : value, quary 
        # 텐서플로가 제공하는 MHA층
        x3=layers.Add()([x2,x])		# 지름길 연결
        x4=layers.LayerNormalization(epsilon=1e-6)(x3)	# 층 정규화
        x5=layers.Dense(d_model*2,activation=tf.nn.gelu)(x4)
        x6=layers.Dropout(0.1)(x5)
        x7=layers.Dense(d_model,activation=tf.nn.gelu)(x6)   
        x8=layers.Dropout(0.1)(x7)        
        x=layers.Add()([x8,x3])		# 지름길 연결
    
        x=layers.LayerNormalization(epsilon=1e-6)(x) # 다음 반복에서 x로 입력하기 때문에 x객체에 저장 
        x=layers.Flatten()(x)
        x=layers.Dropout(0.5)(x)   
        x=layers.Dense(2048,activation=tf.nn.gelu)(x)    
        x=layers.Dropout(0.5)(x)
        x=layers.Dense(1024,activation=tf.nn.gelu)(x)    
        x=layers.Dropout(0.5)(x)    
        output=layers.Dense(n_class,activation='softmax')(x)
        
        model=keras.Model(inputs=input,outputs=output)  # 신경망 내부에 정규화 정보 저장 
        return model

model=create_vit_classifier()
model.layers[1].adapt(x_train)

model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
hist=model.fit(x_train,y_train,
               batch_size=128,
               epochs=100,
               validation_data=(x_test,y_test),
               verbose=1)

res=model.evaluate(x_test,y_test,verbose=0)
print('정확률=',res[1]*100)

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

# 8-2보다 낮은 성능을 보임 
# 정확률= 76.52999758720398