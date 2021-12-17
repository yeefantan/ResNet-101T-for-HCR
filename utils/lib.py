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
import skimage
import random
from itertools import groupby
from keras.preprocessing.sequence import pad_sequences
sys.path.append('../')
from utils.preprocessing import *
from utils.utils import *
from utils.data_augmentation import *
from sklearn.utils import shuffle
