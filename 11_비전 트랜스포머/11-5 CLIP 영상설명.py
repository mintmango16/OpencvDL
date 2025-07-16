from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
os.chdir("11_비전 트랜스포머")

img=Image.open('BSDS_361010.jpg')

processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

captions=['Two horses are running on grass', 'Students are eating', 'Croquet playing on horses', 'Golf playing on horses']
inputs=processor(text=captions,images=img,return_tensors='pt',padding=True)
res=model(**inputs)

import matplotlib.pyplot as plt
plt.imshow(img); plt.xticks([]); plt.yticks([]); plt.show()

logits=res.logits_per_image
probs=logits.softmax(dim=1)
for i in range(len(captions)):
    print(captions[i],': ','{:.2f}'.format(float(probs[0,i]*100.0)))
    
# Two horses are running on grass :  9.45
# Students are eating :  0.00
# Croquet playing on horses :  57.47
# Golf playing on horses :  33.07