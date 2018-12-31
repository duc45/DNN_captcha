import matplotlib.pyplot as plt
import os
import keras.callbacks
import pylab
import cv2
import numpy as np
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model
from random import randint
from utils.utils import decode_predict_ctc, get_img, next_img, decode_batch


home_dir = os.getcwd()
out_put_dirs = home_dir + "/test_captcha_model"
train_data_dir = home_dir + "/train"
test_data_dir = home_dir + "/test"
val_data_dir = home_dir + "/validation"
font_dir = home_dir + "/font/Raleway-Regular.ttf"

img_width, img_height = 150, 50

def predict(run_name, list_img, label_list, indx, model="model.json", weights = "weights.h5", top_paths=1):

	#load model

	json_file = open(os.path.join(out_put_dirs, os.path.join(run_name, model)), "r")
	loaded_model = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model)
	loaded_model.load_weights(os.path.join(out_put_dirs, os.path.join(run_name, weights)))
	print("Loaded model from disk")
	net_inp = loaded_model.get_layer(name="the_input").input
	net_outp = loaded_model.get_layer(name="softmax").output
	test_func = K.function([net_inp], [net_outp])
	count = 0
	#preprocess data
	for index, img in enumerate(list_img):
		if K.image_data_format() == 'channels_first':
			img_dat = np.expand_dims(img, axis=3)
		else:
			img_dat = np.expand_dims(img, axis=0)
		img_dat = img_dat/255
		c = np.expand_dims(np.array(img_dat).T, axis=0)
		top_pred_texts = decode_batch(test_func,c)[0]

		plt.subplot(5, 2, index+1)
		plt.imshow(img, cmap="Greys_r")
		plt.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (label_list[index], top_pred_texts))

	
	fig = pylab.gcf()
	fig.set_size_inches(13, 16)
	plt.savefig(os.path.join(os.path.join(home_dir, "predict"), "predict%02d.png" % (indx)))

list_img, label_list = get_img(test_data_dir, img_width, img_height)


run_name = "build"
predict(run_name, list_img, label_list, 201)
predict(run_name, list_img, label_list, 301, "gru_model.json", "gru_weights.h5")