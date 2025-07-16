import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

(x_train,y_train),(x_test,y_test)=keras.datasets.cifar10.load_data()

n_class=10			# 부류 수
img_siz=(32,32,3)
img_expanded_siz=(72,72,3) # 영상 확대 

patch_siz=6			# 패치 크기
p2=(img_expanded_siz[0]//patch_siz)**2 # 패치 개수
d_model=64			# 임베딩 벡터 차원
h=8				# 헤드 개수
N=6				# 인코더 블록의 개수

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.p_siz=patch_size

    def call(self, img):
        batch_size=tf.shape(img)[0]
        patches=tf.image.extract_patches(images=img,sizes=[1,self.p_siz,self.p_siz,1],strides=[1,self.p_siz,self.p_siz,1],rates=[1,1,1,1],padding="VALID")
        patch_dims=patches.shape[-1]
        patches=tf.reshape(patches,[batch_size,-1,patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self,p2,d_model): # p2: 패치 개수
        super(PatchEncoder,self).__init__()
        self.p2=p2
        self.projection=layers.Dense(units=d_model)
        self.position_embedding=layers.Embedding(input_dim=p2,output_dim=d_model)

    def call(self,patch):
        positions=tf.range(start=0,limit=self.p2,delta=1)
        encoded=self.projection(patch)+self.position_embedding(positions)
        return encoded
    
def create_vit_classifier():
    input=layers.Input(shape=(img_siz))
    nor=layers.Normalization()(input)
    exp=layers.Resizing(img_expanded_siz[0],img_expanded_siz[1])(nor) # 데이터 증강층 적용
    
    x=layers.RandomFlip('horizontal')(exp) # 좌우 반전 적용
    x=layers.RandomRotation(factor=0.02)(x) # 해당 범위에서 랜덤 회전
    x=layers.RandomZoom(height_factor=0.2, width_factor=0.2)(x) # 수평과 수직 방향으로 20% 랜덤 줌아웃 
    
    patches=Patches(patch_siz)(x)		# 패치 생성 : 증강된 영상 입력 받음 
    x=PatchEncoder(p2,d_model)(patches)	# 패치 인코딩

    for _ in range(N):			# 다중 인코더 블록
        x1=layers.LayerNormalization(epsilon=1e-6)(x)		# 층 정규화
        x2=layers.MultiHeadAttention(num_heads=h,key_dim=d_model//h,dropout=0.1)(x1,x1)			# MHA
        x3=layers.Add()([x2,x])		# 지름길 연결
        x4=layers.LayerNormalization(epsilon=1e-6)(x3)	# 층 정규화
        x5=layers.Dense(d_model*2,activation=tf.nn.gelu)(x4)
        x6=layers.Dropout(0.1)(x5)
        x7=layers.Dense(d_model,activation=tf.nn.gelu)(x6)   
        x8=layers.Dropout(0.1)(x7)        
        x=layers.Add()([x8,x3])		# 지름길 연결
    
        x=layers.LayerNormalization(epsilon=1e-6)(x)
        x=layers.Flatten()(x)
        x=layers.Dropout(0.5)(x)   
        x=layers.Dense(2048,activation=tf.nn.gelu)(x)    
        x=layers.Dropout(0.5)(x)
        x=layers.Dense(1024,activation=tf.nn.gelu)(x)    
        x=layers.Dropout(0.5)(x)    
        output=layers.Dense(n_class,activation='softmax')(x)
        
        model=keras.Model(inputs=input,outputs=output)
        return model

model=create_vit_classifier()
model.layers[1].adapt(x_train)

model.compile(optimizer=Adam(),loss=SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
hist=model.fit(x_train,y_train,batch_size=128,epochs=10,validation_data=(x_test,y_test),verbose=1)

res=model.evaluate(x_test,y_test,verbose=0)
print('정확률=',res[1]*100)

import matplotlib.pyplot as plt

import os
os.chdir("11_비전 트랜스포머")

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.savefig('11-2 accuracy_graph.png') # 그래프를 'accuracy_graph.png' 파일로 저장
plt.clf() # 다음 그래프를 위해 현재 그림을 초기화

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.savefig('11-2 loss_graph.png') # 그래프를 'loss_graph.png' 파일로 저장
plt.clf() # 그림 초기화
    
    
# Epoch 10/10
# 391/391 ━━━━━━━━━━━━━━━━━━━━ 10s 25ms/step - accuracy: 0.6208 - loss: 1.0771 - val_accuracy: 0.6810 - val_loss: 0.9086
# 정확률= 68.09999942779541