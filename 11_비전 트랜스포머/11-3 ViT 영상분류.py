from transformers import ViTFeatureExtractor, TFViTForImageClassification
from PIL import Image

import os
os.chdir("11_비전 트랜스포머")

img=[Image.open('BSDS_242078.jpg'),Image.open('BSDS_361010.jpg'),Image.open('BSDS_376001.jpg')]

feature_extractor=ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model=TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs=feature_extractor(img,return_tensors='tf')
res=model(**inputs)

import tensorflow as tf
import matplotlib.pyplot as plt

for i in range(res.logits.shape[0]):
    # 그래프 그리기
    plt.imshow(img[i])
    plt.xticks([])
    plt.yticks([])
    
    # 이미지를 화면에 띄우는 대신 파일로 저장
    plt.savefig(f'result_image_{i}.png')
    print(f"'result_image_{i}.png'로 저장되었습니다.")
    
    # 다음 그래프를 위해 현재 그림을 초기화
    plt.clf()

    # 예측 결과 출력 (기존과 동일)
    predicted_label=int(tf.math.argmax(res.logits[i],axis=-1))
    prob=float(tf.nn.softmax(res.logits[i])[predicted_label]*100.0)
    print(i,'번째 영상의 1순위 부류: ',model.config.id2label[predicted_label], f'{prob:.2f}%')
    
# 'result_image_0.png'로 저장되었습니다.
# 0 번째 영상의 1순위 부류:  umbrella 98.76%
# 'result_image_1.png'로 저장되었습니다.
# 1 번째 영상의 1순위 부류:  croquet ball 3.90%
# 'result_image_2.png'로 저장되었습니다.
# 2 번째 영상의 1순위 부류:  hay 18.51%