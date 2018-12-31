from claptcha import Claptcha
from random import shuffle, choice, randint
from keras import backend as K
import os
import cv2
import numpy as np
import itertools
import tqdm

list_chars = u'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
list_chars_len = len(list_chars)
max_str_len = 16


#### ---- random funct for captcha ---- ####

def rdlen(min, max):
	return randint(min, max)

def randNoise():
	return float(randint(0,100)) / 100

def randFont(FONT_PATH):
	list_font = os.listdir(FONT_PATH)
	rndFont = choice(list_font) 
	return FONT_PATH + "/" + rndFont

#### ---- generate captcha img ---- ####

def next_img(rndLetters, font, im_w, im_h, noise):

	captcha = Claptcha(rndLetters, font, (300, 100), noise = noise)
	text, img = captcha.image
	img_dat = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
	img_dat = cv2.resize(img_dat, (im_w, im_h))
	img_dat = np.array(img_dat) / 255.0
	if K.image_data_format() == 'channels_first':
		img_dat = np.expand_dims(img_dat, axis=3)
	else:
		img_dat = np.expand_dims(img_dat, axis=0)
	img_dat = img_dat.T
	return img_dat

#### ---- convert label ---- ####

def get_label(img):
	label = []
	for c in img:
		label.append(list_chars.find(c))
	for c in range(max_str_len - len(label)):
		label.append(list_chars_len)
	return label

def get_text(label):
	text = []
	for c in label:
		if c >= list_chars_len:
			text.append("")
		elif c < 0:
			text.append("")
		else:
			text.append(list_chars[c])
	return "".join(text)

###### ---- CTC function ---- ######

def ctc_lambda_func(args):
	labels, y_pred, input_length, label_length = args
	y_pred = y_pred[:, 2:, :]
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def decode_batch(test_func, word_batch):
	out = test_func([word_batch])[0]
	ret = []
	for i in range(out.shape[0]):
		out_best = list(np.argmax(out[i, 2:], 1))
		out_best = [k for k, g in itertools.groupby(out_best)]
		out_str = get_text(out_best)
		ret.append(out_str)
	return ret


#### ---- funct for predict ---- ####

def decode_predict_ctc(out, top_paths = 1):
	results = []
	beam_width = 5
	if beam_width < top_paths:
		beam_width = top_paths
	for i in range(top_paths):
		lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
							greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
		text = get_text(lables)
		results.append(text)
	return results
  
def get_img(img_dir, img_width, img_height):
	img_list = []
	label_list = []
	for i in os.listdir(img_dir):
		path = os.path.join(img_dir, i)
		temp = cv2.imread(path, cv2.IMREAD_COLOR)
		img_gray = cv2.cvtColor(np.array(temp), cv2.COLOR_RGB2GRAY)
		img_gray = cv2.resize(img_gray, (img_width,img_height))
		# if K.image_data_format() == 'channels_first':
		# 	img_gray = np.expand_dims(img_gray, axis=3)
		# else:
		# 	img_gray = np.expand_dims(img_gray, axis=0)
		# img_gray = np.array(img_gray).T
		img_list.append(img_gray)
		label_list.append(i.split(".")[0])
	return img_list, label_list


# def data(data_dir):
# 	data = []
# 	for i in tqdm(os.listdir(data_dir)):
# 		path = os.path.join(data_dir, i)
# 		temp = cv2.imread(path, cv2.IMREAD_COLOR)
# 		img = cv2.resize(temp, (img_width,img_height))
# 		img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 		dot = np.array(img_gray)/255.0
# 		if K.image_data_format() == 'channels_first':
# 			dot = np.expand_dims(dot, axis=3)
# 		else:
# 			dot = np.expand_dims(dot, axis=0)
# 		dot = dot.T
# 		label = get_label(i.split(".")[0])
# 		data.append([dot,label,len(i.split(".")[0])])
# 	shuffle(data)
# 	return data