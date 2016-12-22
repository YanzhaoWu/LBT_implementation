# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:47:55 2016

@author: v-yanzwu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#cap = cv2.VideoCapture("D:\\SharedData\\LBT_Train_Set\\2DMOT2015\\train\\TUD-Campus\\TUD-Campus.mp4")
#ret, frame1 = cap.read()
#prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

#hsv = np.zeros_like(frame1)
#hsv[...,1] = 255
#i = 10
#while(i>0):
def generate_optical_flow(frame1, frame2, size):
    img_1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    #print img_1.shape
    img_1 = cv2.resize(img_1, (size[1], size[0])) # ,interpolation=cv2.INTER_CUBIC)
    #print img_1.shape
    #img_2 = img_2[0:img_1.shape[0], 0: img_1.shape[1]]#cv2.resize(img_2, (size[1], size[0]))
    img_2 = cv2.resize(img_2, (size[1], size[0]))
    flow = cv2.calcOpticalFlowFarneback(img_1, img_2, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros(size, dtype = frame1.dtype)
    #print hsv.shape
    hsv[...,1] = 255
    hsv[...,0] = ang * 180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    #print hsv.shape
    #hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)#np.random.randint(1,255,frame1.shape)
    #print hsv.shape
    #print hsv
    
    #plt.imshow(hsv)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr
    #cv2.imshow('frame2',bgr)
    #cv2.imwrite('frame'+ str(i) +'.png', bgr)
    
    #k = cv2.waitKey(30) & 0xff
    #if k == 27:
    #    break
    #elif k == ord('s'):
    #    cv2.imwrite('opticalfb.png',frame2)
    #    cv2.imwrite('opticalhsv.png',bgr)
    #prvs = next
    #i -= 1
    
if __name__ == '__main__':
    img_path =  'D:\\SharedData\\LBT_Train_Set\\train-210\\'
    img1 = cv2.imread(img_path + 'IFWN-sequence-035.avi-00089-R200.png')
    img2 = cv2.imread(img_path + 'IFWN-sequence-035.avi-00096-R200.png')
    processed_img = generate_optical_flow(img1, img2, img1.shape)
    plt.imshow(processed_img)