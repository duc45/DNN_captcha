import random
import string
import os
from PIL import Image
from claptcha import Claptcha

HOME = os.getcwd()
FONT_PATH = HOME + '/font/'
TEMP_FOLDER =  'temp/'
TRAIN_FOLDER = HOME + '/train/'
TEST_FOLDER = HOME + '/test/'
VALID_FOLDER = HOME + '/validation/'
CAPT_LENGTH = 6

def rdLength():
	return random.randint(6,8)

def rdString():
	chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	rndLetters = (random.choice(chars_list) for _ in range(rdLength()))
	return "".join(rndLetters)

def rdFixString(length):
	chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	rndLetters = (random.choice(chars_list) for _ in range(length))
	return "".join(rndLetters)

def randomNoise():
	return float(random.randint(0,100)) / 100

def randomFont():
	list_font = os.listdir(FONT_PATH)
	rndFont = random.choice(list_font) 
	return FONT_PATH + rndFont

def gen_Fix_Captcha(FOLDER, nb_pic, length, img_w, img_h):
	for _ in range(nb_pic):
		''' Fixed length captcha'''
		c = Claptcha(rdFixString(length), randomFont(), (img_w,img_h),
		resample = Image.BILINEAR, noise=0.5)
		if not os.path.exists(FOLDER):
			os.makedirs(FOLDER)
		text, _ =c.write(FOLDER + 'temp.png')
		'''	print(text) '''
		os.rename(FOLDER + 'temp.png',FOLDER + text + '.png')

def gen_Captcha(FOLDER, nb_pic, img_w, img_h):
	for _ in range(nb_pic):
		''' Random length captcha'''
		c = Claptcha(rdString,randomFont(), (img_w,img_h),
		resample = Image.BILINEAR, noise=0)
		if not os.path.exists(FOLDER):
			os.makedirs(FOLDER)
		text, _ =c.write(FOLDER + 'temp.png')
		'''	print(text) '''
		os.rename(FOLDER + 'temp.png',FOLDER + text + '.png')

gen_Fix_Captcha(TEST_FOLDER, nb_pic=16, length=4, img_w=300,img_h=100)
# gen_Fix_Captcha(TEST_FOLDER, nb_pic=18, length=7, img_w=300, img_h=100)
