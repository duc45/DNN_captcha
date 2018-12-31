import matplotlib.pyplot as plt
import os
import keras.callbacks
from keras import backend as K
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Reshape, Dense, Input, Activation
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.utils import plot_model
from utils.utils import ctc_lambda_func
from data_gen.generator import CaptchaGenerator, Visualize_callback


home_dir = os.getcwd()
out_put_dirs = home_dir + "/test_captcha_model"
font_dir = home_dir + "/font"
plot_dir = home_dir + "/plot"

img_width, img_height = 150, 50

#setup parameter

def train(run_name, start_epoch, stop_epoch):
	cnn_act = 'relu'
	cnn_kernel = (3,3)
	cnn_filter = 256
	nb_mpool = 2

	rnn_size = 1024
	kernel_init = 'he_normal'
	batch_sizes = 32
	# nb_train = len(os.listdir(train_data_dir))
	# nb_val = len(os.listdir(val_data_dir))

	nb_train = 16000
	nb_val = 1600

	if K.image_data_format() == 'channels_first':
		input_shape = (1, img_width, img_height)
	else:
		input_shape = (img_width, img_height, 1)
	
	# ----- data gen -----

 	
	data_gen = CaptchaGenerator(ds_factor=2**2,
 						  im_w = img_width,
 						  im_h = img_height,
 						  batch_size = batch_sizes,
 						  font_dir = font_dir,
 						  nb_train = nb_train,
 						  nb_val = nb_val)


	# Input 150x50 color image, using 32 filter size 3x3
	Input_data = Input(name='the_input', shape=input_shape, dtype='float32')
	cnn = Conv2D(64, cnn_kernel, activation=cnn_act, padding='same',
                kernel_initializer=kernel_init, name='conv1')(Input_data)
	cnn = MaxPooling2D(pool_size=(2, 2), strides=2, name='max1')(cnn)

	cnn = Conv2D(128, cnn_kernel, activation=cnn_act, padding='same',
				kernel_initializer=kernel_init, name='conv2')(cnn)
	cnn = Conv2D(cnn_filter, cnn_kernel, activation=cnn_act, padding='same',
    			kernel_initializer=kernel_init, name='conv3')(cnn)
	
	cnn = MaxPooling2D(pool_size=(2,2), strides=2)(cnn)
	
	conv_to_rnn_dims = (img_width//(2**nb_mpool), (img_height // (2**nb_mpool))*cnn_filter)
	rnn = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(cnn)
	
	lstm_1a = LSTM(rnn_size, return_sequences=True, kernel_initializer=kernel_init, name='gru_1a')(rnn)
	lstm_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer=kernel_init, name='gru_1b')(rnn)
	lstm1 = concatenate([lstm_1a, lstm_1b])
	lstm_2a = LSTM(rnn_size, return_sequences=True, kernel_initializer=kernel_init, name='gru_2a')(lstm1)
	lstm_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer=kernel_init, name='gru_2b')(lstm1)


	rnn = Dense(data_gen.get_output_size(), kernel_initializer=kernel_init
          ,name='dense2')(concatenate([lstm_2a, lstm_2b]))
	y_pred = Activation('softmax', name='softmax')(rnn)
	#Model(inputs=Input_data, outputs=y_pred).summary()

	labels = Input(name='the_labels', shape=[data_gen.max_str_leng], dtype='float32')
	input_length = Input(name='input_length', shape=[1], dtype='int64')
	label_length = Input(name='label_length', shape=[1], dtype='int64')

	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])

	sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)


	model = Model(inputs=[Input_data, labels, input_length, label_length], outputs=loss_out)
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=['accuracy'])

	if start_epoch > 0:
		weights_file = os.path.join(out_put_dirs, os.path.join(run_name, 'weights.h5'))
		model.load_weights(weights_file)

	test_func = K.function([Input_data], [y_pred])

	viz_cb = Visualize_callback(run_name, out_put_dirs, test_func, data_gen.next_val())

	history = model.fit_generator(generator = data_gen.next_train(),
						steps_per_epoch = (nb_train//batch_sizes),
						epochs=stop_epoch,
						validation_data = data_gen.next_val(),
						validation_steps = (nb_val//batch_sizes),
						callbacks=[viz_cb, data_gen],
						initial_epoch=start_epoch)
	
	# plot accuracy and loss training process

	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(plot_dir + "/accuracy.png")
	plt.close()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(plot_dir + "/loss.png")
	plt.close()

	model_pred = Model(inputs=Input_data, outputs=y_pred)
	model_json = model_pred.to_json()
	with open(out_put_dirs + "/" + run_name + "/model.json", "w") as json_file:
		json_file.write(model_json)

run_name = "build"
train(run_name,0,1)

