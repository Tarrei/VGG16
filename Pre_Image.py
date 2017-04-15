 #!/usr/bin/env python
# -*- coding:utf8 -*-
#Author: Zhang Jingbin
from PIL import Image
import sys
import os # process string
import glob # find files
from pandas import Series, DataFrame
# from keras.preprocessing import image
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

def pre_image(path, image_size):
	# image_vector_len = np.prod(image_size)
	# print image_vector_len
	files = glob.glob(path)
	images = []
	for fl in files:
		img = Image.open(fl)
		if img == None:
			pass
		else:
			# im = cv2.resize(img, image_size).astype(np.float32)
			img = img.resize(image_size) 
			im = np.asarray(img, dtype='float32')
			im = im.transpose(2,0,1)
			# print im.shape
			images.append(im)
	images = np.asarray(images, dtype='float32')
	images = preprocess_input(images) 
	return images

if __name__ == '__main__':
	path = os.path.join('imgs', '*.jpg')
	images = pre_image(path, (224, 224))
	print images 
