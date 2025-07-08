import tensorflow as tf
from tensorflow.keras import datasets 
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=datasets.mnist.load_data() # 필기 숫자 데이터셋
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
plt.figure(figsize=(24,3))
plt.suptitle('MNIST',fontsize=30)
for i in range(10): # 모양 확인 및 샘플 10개 출력 
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i],cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.title(str(y_train[i]),fontsize=30)
plt.show()

(x_train ,y_train),(x_test, y_test)=datasets.cifar10.load_data() # 자연 영상 데이터셋 
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
class_names=['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']
plt.figure(figsize=(24,3))
plt.suptitle('CIFAR-10',fontsize=30)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[y_train[i,0]],fontsize=30)
plt.show()
    
# 딥러닝에서의 다차원 배열 = tensor

# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
# 특징 벡터는 숫자를 28*28 맵으로 표현하고, 참값은 숫자 부류를 나타내기 위해 0-9 사이의 값을 가짐 

# (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
# 특징 벡터는 숫자를 32*32*3 맵으로 표현하고, RGB로 인한 3장 구조 -> 부류가 10개 이므로 참값 0-9의 값 가짐 
