# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 22:36:17 2016

@author: v-yanzwu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("D:\\SharedData\\LBT_Train_Set\\2DMOT2015\\train\\TUD-Campus\\TUD-Campus.mp4")
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)
hsv[...,1] = 255
i = 10
while(i>0):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    print frame2.shape
    flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    #cv2.imshow('frame2',bgr)
    #cv2.imwrite('frame'+ str(i) +'.png', bgr)
    
    #k = cv2.waitKey(30) & 0xff
    #if k == 27:
    #    break
    #elif k == ord('s'):
    #    cv2.imwrite('opticalfb.png',frame2)
    #    cv2.imwrite('opticalhsv.png',bgr)
    prvs = next
    i -= 1
'''
cap.release()
cv2.destroyAllWindows()
'''