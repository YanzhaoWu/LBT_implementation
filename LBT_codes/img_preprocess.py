# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:44:45 2016

@author: v-yanzwu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization

def planeFitFunc(params, xdata, ydata):
    return (ydata - np.dot(xdata, params[0:2]) - np.full(ydata.shape[0],params[2],dtype=np.float))

def planeFitOneLayer(img):
    xdata = np.zeros([img.shape[0] * img.shape[1], 2])
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            xdata[i * img.shape[1] + j][0] = i #y
            xdata[i * img.shape[1] + j][1] = j #x
    
    ydata = img.flatten()
    x0 = np.array([0,0,0])
    parameters = optimization.leastsq(planeFitFunc, x0, args=(xdata, ydata))
    params = parameters[0]

    ydata = (ydata - np.dot(xdata, params[0:2]) - np.full(ydata.shape[0],params[2],dtype=np.float))

    new_img = ydata.reshape(img.shape)
    return new_img

def img_preprocess_pre(img):
    #result = np.zeros(img.shape)
    #for layer in range(0, img.shape[2]):
        # Histograms Equalization in OpenCV
    #layer = 0
    tmp_img = cv2.equalizeHist(img)
    tmp_img = planeFitOneLayer(tmp_img)
    return tmp_img

def img_preprocess_func(img):
    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    #print img_luv.shape
    processed_img_luv = img_luv.copy()
    processed_img_luv[:, :, 0] = img_preprocess_pre(img_luv[:,:, 0])
    processed_img = cv2.cvtColor(processed_img_luv, cv2.COLOR_LUV2BGR)
    return processed_img
    #plt.imshow(processed_img, cmap = plt.cm.gray)

    
    
if __name__ == '__main__':
    caffe_root =  'D:\\Projects\\caffe\\'
    img = cv2.imread(caffe_root + 'examples\\images\\cat.jpg')
    #plt.imshow(img)
    #img_luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    #print img_luv.shape
    #processed_img_luv = img_luv.copy()
    #processed_img_luv[:, :, 0] = img_preprocess_pre(img_luv[:,:, 0])
    #processed_img = cv2.cvtColor(processed_img_luv, cv2.COLOR_LUV2BGR)
    processed_img = img_preprocess_func(img)
    res = np.hstack((img, processed_img))
    plt.imshow(res)
    
#equ = cv2.equalizeHist(img[:,:,1])
#res = np.hstack((img[:,:,1],equ)) #stacking images side-by-side
#cv2.imwrite('res.png',res)


#A plane fitting






#contrast = np.hstack((img[:,:,1],equ,new_img))
#plt.imshow(contrast,cmap = plt.cm.gray)

# Caffe part

#import caffe

#net = caffe.Net("D:\\SharedData\\LBT\\LBT_deploy.prototxt", caffe.TEST)
#net.blobs['data'].data[...] = np.random.randint(0, 255, (1, 10, 121, 53))
#net.blobs['data'].reshape(img.shape)
#out = net.forward()

#gb_input1 = out['fc6'].copy()
#print gb_input1.shape
#print gb_input1