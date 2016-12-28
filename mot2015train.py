# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:21:06 2016

@author: v-yanzwu
"""

import numpy as np
import os
import cv2

from optical_flow_generator import *
from img_preprocess import *
train_data_path = 'D:\\SharedData\\LBT_Train_Set\\2DMOT2015\\train\\'

#train_data_set = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 
train_data_set = ['KITTI-16']#, 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
train_data_set_num = len(train_data_set)

train_data_generate_num_ground_truth_positive = 1000
train_data_generate_num_ground_truth_negative = 1000
train_data_generate_num_detection_negative = 1000
    

for train_data_set_id in range(0, train_data_set_num):
    #ground_truth_boxes = np.loadtxt(train_data_path + train_data_set[train_data_set_id] + '\\gt\\gt.txt', delimiter = ',')
    #ground_truth_boxes_num = ground_truth_boxes.shape[0]
    #frame_num = int(ground_truth_boxes[ground_truth_boxes_num - 1][0])
    
    # Optical flow process
    if not os.path.exists(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow'):
        os.mkdir(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow')
    if not os.path.exists(train_data_path + train_data_set[train_data_set_id] + '\\preprossed'):
        os.mkdir(train_data_path + train_data_set[train_data_set_id] + '\\preprossed')
    for frame_idx in range(1, 209):
        img1 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\img1\\' + ('%06d' % frame_idx) + '.jpg')
        img2 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\img1\\' + ('%06d' % (frame_idx + 1)) + '.jpg')
        optical_flow_img = generate_optical_flow(img1, img2, img1.shape)
        cv2.imwrite(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow\\' +  ('%06d' % frame_idx) + '.jpg', optical_flow_img)
        img1_preprocessed = img_preprocess_func(img1)
        cv2.imwrite(train_data_path + train_data_set[train_data_set_id] + '\\preprossed\\' + ('%06d' % frame_idx) + '.jpg', img1_preprocessed)
'''
    ground_truth_positive = train_data_generate_num_ground_truth_positive
    ground_truth_negative = train_data_generate_num_ground_truth_negative
    while (ground_truth_positive > 0) and (ground_truth_negative > 0):
        random_1 = np.random.randint(0, ground_truth_boxes_num)
        random_2 = np.random.randint(0, ground_truth_boxes_num)
'''    
        

