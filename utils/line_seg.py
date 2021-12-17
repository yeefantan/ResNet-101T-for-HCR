import cv2
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from scipy.signal import argrelmin

import os
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
sns.set(rc={'figure.figsize' : (22, 10)})
sns.set_style("darkgrid", {'axes.grid' : True})

def crop_text_to_lines(text, blanks):
    x1 = 0
    y = 0
    lines = []
    for i, blank in enumerate(blanks):
        x2 = blank
        print("x1=", x1, ", x2=", x2, ", Diff= ", x2-x1)
        line = text[:, x1:x2]
        lines.append(line)
        x1 = blank
    return lines

def display_lines(lines_arr, orient='vertical'):
    plt.figure(figsize=(30, 30))
    if not orient in ['vertical', 'horizontal']:
        raise ValueError("Orientation is on of 'vertical', 'horizontal', defaul = 'vertical'") 
    if orient == 'vertical': 
        for i, l in enumerate(lines_arr):
            line = l
            plt.subplot(2, 10, i+1)  # A grid of 2 rows x 10 columns
            plt.axis('off')
            plt.title("Line #{0}".format(i))
            _ = plt.imshow(line, cmap='gray', interpolation = 'bicubic')
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    else:
            for i, l in enumerate(lines_arr):
                line = l
                plt.subplot(40, 1, i+1)  # A grid of 40 rows x 1 columns
                plt.axis('off')
                plt.title("Line #{0}".format(i))
                _ = plt.imshow(line, cmap='gray', interpolation = 'bicubic')
                plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
def smooth(x, window_len=11, window='hanning'):
#     if x.ndim != 1:
#         raise ValueError("smooth only accepts 1 dimension arrays.") 
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.") 
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'") 
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y

def createKernel(kernelSize, sigma, theta):
    "create anisotropic filter kernel according to given parameters"
    assert kernelSize % 2 # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
            xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
            yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

def applySummFunctin(img):
    res = np.sum(img, axis = 0)    #  summ elements in columns
    return res

def transpose_lines(lines):
    res = []
    for l in lines:
        line = np.transpose(l)
        res.append(line)
    return res

def showImg(img, cmap=None):
    plt.imshow(img, cmap=cmap, interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def normalize(img):
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img

def load_line_removal_data(datadir,code):
    
    datalength = 0
    data = list()
    labels = list()
    
    path_list = os.listdir(datadir)
    if('.DS_Store') in path_list:
        path_list.remove('.DS_Store')
        
    for img in path_list:
        img_ = os.path.join(datadir,img)
        if img[0] == code:
            #img_ = cv2.imread(img_)
            #img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)
            data.append(img)
            
    return np.asarray(data)

def save(img,name,code):
    direc = code
    cv2.imwrite('Segment/'+str(code)+'/'+name+'.jpg',np.asarray(img))


def main():
    kernelSize=25
    sigma=12
    theta=3
    data = list()
    data = load_line_removal_data("Line Removal",'j')
    for i in range(len(data)):
        code = data[0][0]
        img = "Line Removal/"+str(data[i])
        img_ = cv2.imread(img)

        img_2 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        img_3 = np.transpose(img_2)
        k = createKernel(4, sigma, theta)
        imgFiltered = cv2.filter2D(img_3, -1, k, borderType=cv2.BORDER_REPLICATE)
        img_4 = normalize(imgFiltered)
        summ3 = applySummFunctin(img_4)
        smoothed3 = smooth(summ3, 35)
        mins3 = argrelmin(smoothed3, order=2)
        arr_mins3 = np.array(mins3)
        found_lines3 = crop_text_to_lines(img_3, arr_mins3[0])
        res_lines3 = transpose_lines(found_lines3)
        j = 0
        for segment in range(len(res_lines3)):
            if res_lines3[segment].shape[0]!=0:
                img_segment = trim(Image.fromarray(res_lines3[segment]))
                if img_segment is not None:
                    j+=1
                    name = data[i][:-4] + '_' + str("{0:03}".format(j))
                    save(img_segment,name,code)