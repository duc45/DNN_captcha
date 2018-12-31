import os
import editdistance
import pylab
import matplotlib.pyplot as plt
import numpy as np
import keras.callbacks
from random import choice
from utils.utils import rdlen, randNoise, randFont, next_img, get_label, get_text, decode_batch
from keras import backend as K

list_chars = u'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
list_chars_len = len(list_chars)

class CaptchaGenerator(keras.callbacks.Callback):
	def __init__(self, ds_factor,im_w, im_h,batch_size, font_dir, max_str_leng=16, nb_train=16000, nb_val=1600):
	
		super(CaptchaGenerator, self).__init__()
		self.batch_size= batch_size
		self.im_w = im_w
		self.im_h = im_h
		self.font_dir = font_dir
		self.max_str_leng = max_str_leng
		self.ds_factor = ds_factor
		self.cur_train_index = 0
		self.cur_val_index = 0
		self.nb_train = nb_train
		self.nb_val = nb_val

	def get_output_size(self):
		return list_chars_len + 1

	def build_captcha(self, nb_train, nb_val, im_w, im_h,
		rdNoise=False, noise=0.0 , fix_len=True, min_str_len=6, max_str_len=8, rdFont=False):
		
		font = self.font_dir + "/Raleway-Regular.ttf"

		train_X =[]
		train_Y = []
		train_Y_len = []
		split_tr = nb_train//11
		split_char = nb_train//list_chars_len
		for i in range(nb_train):
			first_letter = ""
			for indx in range(0,list_chars_len):
				if i>=indx*split_char:
					first_letter = "".join(list_chars[indx])
			rndLetters = first_letter

			if fix_len:
				rndLetters = rndLetters.join(choice(list_chars) for _ in range(min_str_len))
			else:
				rndLetters = rndLetters.join(choice(list_chars) for _ in range(rdlen(min_str_len, max_str_len)))
					
			if i < nb_val-5:
				rndLetters = " "

			if rdFont:
				font = randFont(self.font_dir)
			
			if rdNoise:
				noise = randNoise()
			else:
				noise = 0.1
				for tem in range(0,10):
					if i>=tem*split_tr:
						noise=noise*tem

			img_dat = next_img(rndLetters=rndLetters, font=font, im_w=im_w, im_h=im_h, noise=noise)
			
			train_Y.append(get_label(rndLetters))
			train_Y_len.append(len(rndLetters))
			train_X.append(img_dat)

		self.train_X = np.array(train_X)
		self.train_Y = np.array(train_Y)
		self.train_Y_len = np.array(train_Y_len)

		val_X =[]
		val_Y = []
		val_Y_len = []
		split_val = nb_val//11
		split_char = nb_val//list_chars_len
		for i in range(nb_val):

			first_letter = ""
			for indx in range(0,list_chars_len):
				if i>=indx*split_char:
					first_letter = "".join(list_chars[indx])

			rndLetters=first_letter

			if fix_len:
				rndLetters = rndLetters.join(choice(list_chars) for _ in range(min_str_len-1))
			else:
				rndLetters = rndLetters.join(choice(list_chars) for _ in range(rdlen(min_str_len-1, max_str_len-1)))
			
			if i<nb_val-5:
				rndLetters = " "


			if rdNoise:
				noise = randNoise()
			else:
				noise = 0.1
				for tem in range(0,10):
					if i>=tem*split_val:
						noise=noise*tem	

			if rdFont:
				font = randFont(self.font_dir)

			img_dat = next_img(rndLetters=rndLetters, font=font, im_w=im_w, im_h=im_h, noise=noise)
			
			val_Y.append(get_label(rndLetters))
			val_Y_len.append(len(rndLetters))
			val_X.append(img_dat)

		self.val_X = np.array(val_X)
		self.val_Y = np.array(val_Y)
		self.val_Y_len = np.array(val_Y_len)

	def get_batch(self, index, size, train):
		X_data = []
		labels = []
		input_length = np.zeros([size, 1])
		label_length = np.zeros([size, 1])
		source_str = []
		for i in range(size):
			if train:
				X_data.append(self.train_X[index + i])
				labels.append(self.train_Y[index + i])
				input_length[i] = self.im_w // self.ds_factor - 2
				label_length[i] = self.train_Y_len[index + i]
				source_str.append(get_text(labels[i]))
			else:
				X_data.append(self.val_X[index + i])
				labels.append(self.val_Y[index + i])
				input_length[i] = self.im_w // self.ds_factor - 2
				label_length[i] = self.val_Y_len[index + i]
				source_str.append(get_text(labels[i]))
		X_data = np.array(X_data)
		labels = np.array(labels)
		inputs = {
			'the_input': X_data,
			'the_labels': labels,
			'input_length': input_length,
			'label_length': label_length,
			'source_str': source_str
			}
		outputs = {'ctc': np.zeros([size])}
		return (inputs, outputs)

	def next_train(self):
		while 1:
			ret = self.get_batch(self.cur_train_index,
                                 min(self.batch_size, self.nb_train - self.cur_train_index),
                                 train=True)
			self.cur_train_index += self.batch_size
			if self.cur_train_index >= self.nb_train:
				self.cur_train_index = 0
			yield ret

	def next_val(self):
		while 1:
			ret = self.get_batch(self.cur_val_index,
                                 min(self.batch_size, self.nb_val - self.cur_val_index),
                                 train=False)
			self.cur_val_index += self.batch_size
			if self.cur_val_index >= self.nb_val:
				self.cur_val_index = 0
			yield ret

	def on_train_begin(self, logs={}):
		self.build_captcha(self.nb_train, self.nb_val, self.im_w, self.im_h, rdNoise = True, fix_len=False)


class Visualize_callback(keras.callbacks.Callback):
	def __init__(self, run_name, out_put_dirs, test_func, data_gen, num_display_words=6):
		self.test_func = test_func
		self.output_dir = os.path.join(out_put_dirs, run_name)
		self.data_gen = data_gen
		self.num_display_words = num_display_words
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)
   
	def show_edit_distance(self, num):
		num_left = num
		mean_norm_ed = 0.0
		mean_ed = 0.0
		while num_left > 0:
			word_batch = next(self.data_gen)[0]
			num_proc=min(word_batch['the_input'].shape[0],num_left)
			decoded_res= decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
			for j in range(num_proc):
				edit_dist =editdistance.eval(decoded_res[j], word_batch['source_str'][j])
				mean_ed += float(edit_dist)
				mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
			num_left -= num_proc
		mean_norm_ed = mean_norm_ed/num
		mean_ed = mean_ed / num
		print('\nOut of %d samples: Mean edit distance: %0.3f Mean normalized edit distance: %0.3f' % (num, mean_ed, mean_norm_ed))

	def on_epoch_end(self, epoch, logs={}):
		self.model.save_weights(os.path.join(self.output_dir, 'weights.h5'))
		self.show_edit_distance(256)
		word_batch = next(self.data_gen)[0]
		res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
		if word_batch['the_input'][0].shape[0] < 256:
			cols = 2
		else:
			cols = 1
		for i in range(self.num_display_words):
			plt.subplot(self.num_display_words // cols, cols, i+1)
			if K.image_data_format() == 'channels_first':
				the_input = word_batch['the_input'][i, 0, :, :]
			else:
				the_input = word_batch['the_input'][i, :, :, 0]
			plt.imshow(the_input.T, cmap='Greys_r')
			plt.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
		fig = pylab.gcf()
		fig.set_size_inches(10, 13)
		plt.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
		plt.close()




# class DataGenerator(keras.callbacks.Callback):
# 	def __init__(self, train_dir, ds_factor,train_size,
# 		im_w, img_h, val_dir, batch_size, val_size, max_str_leng=16):
		
# 		super(DataGenerator, self).__init__()
# 		self.train_dir = train_dir
# 		self.val_dir = val_dir
# 		self.batch_size= batch_size
# 		self.train_size = train_size
# 		self.im_w = im_w
# 		self.img_h = img_h
# 		self.val_size =val_size
# 		self.max_str_leng = max_str_leng
# 		self.ds_factor = ds_factor
# 		self.cur_train_index = 0
# 		self.cur_val_index = 0

# 	def get_output_size(self):
# 		return list_chars_len + 1

# 	def get_data(self):
# 		train_data = data(self.train_dir)
# 		train_X =[]
# 		train_Y = []
# 		train_Y_len = []
# 		for i in train_data:
# 			# train_X.append(np.expand_dims(i[0], axis=3).T)
# 			train_X.append(i[0])
# 			train_Y.append(i[1])
# 			train_Y_len.append(i[2])
# 		self.train_X = np.array(train_X)
# 		self.train_Y = np.array(train_Y)
# 		self.train_Y_len = np.array(train_Y_len)

# 		#self.train_X = np.array(i[0] for i in train_data)
# 		#self.train_Y = np.array(i[1] for i in train_data)
# 		#self.train_Y_len = np.array(i[2]for i in train_data)

# 		val_data = data(self.val_dir)
# 		val_X = []
# 		val_Y = []
# 		val_Y_len = []
# 		for i in val_data:
# 			# val_X.append(np.expand_dims(i[0], axis=3).T)
# 			val_X.append(i[0])
# 			val_Y.append(i[1])
# 			val_Y_len.append(i[2])
# 		self.val_X = np.array(val_X)
# 		self.val_Y = np.array(val_Y)
# 		self.val_Y_len = np.array(val_Y_len)

# 		#self.val_X = np.array(i[0] for i in val_data)
# 		#self.val_Y = np.array(i[1] for i in val_data)
# 		#self.val_Y_len = np.array(i[2] for i in val_data)


# 	def get_batch(self, index, size, train):
# 		X_data = []
# 		labels = []
# 		input_length = np.zeros([size, 1])
# 		label_length = np.zeros([size, 1])
# 		source_str = []
# 		for i in range(size):
# 			if train:
# 				X_data.append(self.train_X[index + i])
# 				labels.append(self.train_Y[index + i])
# 				input_length[i] = self.im_w // self.ds_factor - 2
# 				label_length[i] = self.train_Y_len[index + i]
# 				source_str.append(get_text(labels[i]))
# 			else:
# 				X_data.append(self.val_X[index + i])
# 				labels.append(self.val_Y[index + i])
# 				input_length[i] = self.im_w // self.ds_factor - 2
# 				label_length[i] = self.val_Y_len[index + i]
# 				source_str.append(get_text(labels[i]))
# 		X_data = np.array(X_data)
# 		labels = np.array(labels)
# 		inputs = {
# 			'the_input': X_data,
# 			'the_labels': labels,
# 			'input_length': input_length,
# 			'label_length': label_length,
# 			'source_str': source_str
# 			}
# 		outputs = {'ctc': np.zeros([size])}
# 		return (inputs, outputs)

# 	def next_train(self):
# 		while 1:
# 			ret = self.get_batch(self.cur_train_index,
#                                  min(self.batch_size, self.train_size - self.cur_train_index),
#                                  train=True)
# 			self.cur_train_index += self.batch_size
# 			if self.cur_train_index >= self.train_size:
# 				self.cur_train_index = 0
# 			yield ret

# 	def next_val(self):
# 		while 1:
# 			ret = self.get_batch(self.cur_val_index,
#                                  min(self.batch_size, self.val_size - self.cur_val_index),
#                                  train=False)
# 			self.cur_val_index += self.batch_size
# 			if self.cur_val_index >= self.val_size:
# 				self.cur_val_index = 0
# 			yield ret

# 	def on_train_begin(self, logs={}):
# 		self.get_data()
