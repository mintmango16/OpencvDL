import tensorflow.keras.datasets as ds
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 데이터 증강 지원 클래스 
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=ds.cifar10.load_data()
x_train=x_train.astype('float32'); x_train/=255 # 실수형 변환 
x_train=x_train[0:15,]; y_train=y_train[0:15,]	# 앞 15개에 대해서만 증대 적용
class_names=['airplane','automobile','bird','cat','deer','dog','flog','horse','ship','truck']

plt.figure(figsize=(20,2)) # 15개 영상 디스플레이 
plt.suptitle("First 15 images in the train set")
for i in range(15):
    plt.subplot(1,15,i+1)
    plt.imshow(x_train[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[int(y_train[i])])
plt.show()    

batch_siz=4			# 증강을 통해 한 번에 생성하는 양(미니 배치 크기)
# 증강 : 변환 방식과 범위 지정
generator=ImageDataGenerator(rotation_range=20.0, # 회전 각도 범위
                             width_shift_range=0.2, # 가로 방향 이동 비율
                             height_shift_range=0.2, # 세로 방향 이동 
                             horizontal_flip=True) #좌우 반전 시도 
gen=generator.flow(x_train, # x_train 에서 증강 지시 
                   y_train,
                   batch_size=batch_siz) 

for a in range(3): # 데이터 생성 
    img,label= next(gen)	# 미니 배치만큼 생성 : 호출 마다 데이터 증강 및 생성 -> 생성 영상과 참값 저장 
    plt.figure(figsize=(8,2.4))
    plt.suptitle("Generatior trial "+str(a+1))
    for i in range(batch_siz): # 증강도니 영상 네장 디스플레이 
        plt.subplot(1,batch_siz,i+1)
        plt.imshow(img[i])
        plt.xticks([]); plt.yticks([])
        plt.title(class_names[int(label[i])])
    plt.show()