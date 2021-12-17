import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy import ndimage
import sys
sys.path.append('../')
from utils.line_seg import *
from utils.line_removal import *
from natsort import natsorted


def load_datargb(datadir,categories):
    
    datalength = 0
    data = list()
    labels = list()
    full_name = list()
    for i,category in enumerate(categories):
        path = os.path.join(datadir,category)
        path_list = os.listdir(path)
        if('.DS_Store') in path_list:
            path_list.remove('.DS_Store')
        for img in natsorted(path_list):
            img_ = os.path.join(path,img)
            img_ = cv2.imread(img_)
            img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)
            data.append(img_)
        full_name.extend(natsorted(path_list))
            
    return data,full_name

def load_data(datadir,categories):
    
    datalength = 0
    data = list()
    labels = list()
    full_name = list()
    for i,category in enumerate(categories):
        path = os.path.join(datadir,category)
        path_list = os.listdir(path)
        if('.DS_Store') in path_list:
            path_list.remove('.DS_Store')
        for img in natsorted(path_list):
            img_ = os.path.join(path,img)
            img_ = cv2.imread(img_)
            img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
            data.append(img_)
        full_name.extend(natsorted(path_list))
            
    return data,full_name

def load_data_augmented(datadir,categories):
    
    datalength = 0
    data = list()
    labels = list()
    full_name = list()
    for i,category in enumerate(categories):
        path = os.path.join(datadir,category)
        path_list = os.listdir(path)
        if('.DS_Store') in path_list:
            path_list.remove('.DS_Store')
        for img in range(len(path_list)):
            img_name = path+'/'+str(img)+'.jpg'
            img_ = cv2.imread(img_name)
            img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
            data.append(img_)
        full_name.extend(sorted(path_list))
            
    return data,full_name

def read_img(datadir):

	img = cv2.imread(datadir)
	img_ = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	return img

def height_width_adjustment(data):
	max_height = 48
	max_width = 4000
	img = data

	trans_ratio = max_height/data.shape[0]
	trans_ratio = int(trans_ratio)

	if trans_ratio<1:
		width = data.shape[1]
		height = data.shape[0]
	else:
		width = data.shape[1]*trans_ratio
		height = max_height

	image = Image.fromarray(data)
	image_resized = image.resize((width,height))


	current_width = np.asarray(data).shape[1]
	gap = max_width - current_width
	im = Image.fromarray(data)
	im_new = add_padding(im, 0, gap, 0, 0, (255, 255, 255))
	return im_new

def add_padding(img,top,right,bottom,left,color):
    width = np.asarray(img).shape[1]
    height = np.asarray(img).shape[0]
    new_width = width+right+left
    new_height = height+top+bottom
    new_img = Image.new(img.mode,(new_width,new_height),color)
    new_img.paste(img,(left,top))
    return new_img

def automate_preprocess_line_img(datadir):
	img = read_img(datadir)
	img = height_width_adjustment(img)
	return img