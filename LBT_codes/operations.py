# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

caffe_root =  'D:\\Projects\\caffe\\'



# Histograms Equalization in OpenCV
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(caffe_root + 'examples\\images\\cat_gray.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
#cv2.imwrite('res.png',res)


#A plane fitting
import scipy.optimize as optimization



def planeFitFunc(params, xdata, ydata):
    return (ydata - np.dot(xdata, params[0:2]) - np.full(ydata.shape[0],params[2],dtype=np.float))

xdata = np.zeros([equ.shape[0]*equ.shape[1], 2])
    
for i in range(0, equ.shape[0]):
    for j in range(0, equ.shape[1]):
        xdata[i * equ.shape[1] + j][0] = i #y
        xdata[i * equ.shape[1] + j][1] = j #x
        

ydata = equ.flatten()
x0 = np.array([0,0,0])

parameters = optimization.leastsq(planeFitFunc, x0, args=(xdata, ydata))
print parameters

params = parameters[0]

ydata = (ydata - np.dot(xdata, params[0:2]) - np.full(ydata.shape[0],params[2],dtype=np.float))

new_img = ydata.reshape(equ.shape)
contrast = np.hstack((img,equ,new_img))
plt.imshow(contrast)