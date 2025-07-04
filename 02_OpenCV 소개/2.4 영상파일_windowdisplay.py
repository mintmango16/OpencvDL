import cv2 as cv
import sys

img = cv.imread('hangang.jpg') # 영상 읽기

if img is None : 
    sys.exit('no file')
    
cv.imshow('Image Display',img)

cv.waitKey()
cv.destroyAllWindows()


# type(img)
# Out[7]: numpy.ndarray

# img.shape
# Out[8]: (506, 875, 3)  : 이미지는 3차원 배열임 blue, green, red에 해당 (BDR)


