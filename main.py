 #!/usr/bin/env python
# -*- coding:utf8 -*-
#Author: Zhang Jingbin
import numpy as np
import os
import sys 
import glob
import time
from Pre_Image import pre_image
from VGG16_Model import VGG_16
from keras.optimizers import SGD 

if __name__ == '__main__':

	t0 = time.time()
	model = VGG_16('vgg16_weights.h5')
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')

	path = os.path.join('imgs', '*.jpg')
	images = pre_image(path, (224, 224))
	y_pred = model.predict(images)
	print('Test Result:', decode_predictions(y_pred))  
	# 输出5个最高概率：(类名, 语义概念, 预测概率)
	print("Time Cost: %.2f seconds" % (time.time() - t0))