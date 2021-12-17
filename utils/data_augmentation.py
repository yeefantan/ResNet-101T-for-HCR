import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,BatchNormalization,Lambda,Bidirectional,LSTM, ZeroPadding2D, GlobalAveragePooling2D, Dense, Conv2D, Convolution2D, Flatten, Dropout, MaxPool2D, MaxPooling2D, Activation
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import cv2
import matplotlib as plt
import tqdm
import os
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras import applications
from keras.models import Model
from keras.optimizers import SGD, Adagrad, Adam, Adadelta, RMSprop
from tensorflow import keras
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
import sys
import copy
import unicodedata
import string
from itertools import groupby
from keras.preprocessing.sequence import pad_sequences
sys.path.append('../')
from utils.preprocessing import *
from utils.utils import *

import skimage
import random

def reduce_line(img):
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

def noise(img):
    return skimage.util.random_noise(img, mode='gaussian', clip=True)

def blur(img):
    return cv2.blur(img,(8,8))


def expand_width(img):
    pxmin = np.min(img)
    pxmax = np.max(img)
    imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)
    return imgMorph

def augmentation(img):
    augmented = list()
    for i in img:
        augmented.append(i)
        augmented.append(reduce_line(i))
        augmented.append(noise(i)*255)
        augmented.append(blur(i))
        augmented.append(expand_width(i))
        
    return np.asarray(augmented)

def save_augmented(imgs):
    for i in range(len(imgs)):
        width = 800
        height = 48
        width = int(width)
        height = int(height)
        image = Image.fromarray(test1[i])
        image = image.convert('RGB')
        image_resized = image.resize((width,height))
        image_resized.save('data_augmented5/'+str(i)+'.jpg','PNG', quality=100)