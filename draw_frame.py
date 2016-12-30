# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:00:34 2016

@author: v-yanzwu
"""

import numpy as np
import cv2
#overlap_threshold = 0.6
def get_test_locations(frame_idx, test_boxes):
    i = 0
    is_first_time = True
    result = None
    while (i < test_boxes.shape[0]) and (test_boxes[i][0] <= frame_idx):
        #print i
        if (test_boxes[i][0] == frame_idx):
            if is_first_time:
                result = np.array([test_boxes[i][0:6]])
                #print result
                is_first_time = False
            else:
                result = np.vstack((result, test_boxes[i][0:6]))
                #print result
        i += 1
    return result

data_set_root = 'D:\\SharedData\\LBT_Train_Set\\2DMOT2015\\test\\'
data_set_name = 'KITTI-16'
test_result_boxes = np.loadtxt('KITTI-16small' + '.txt', delimiter = ',')
test_result_boxes_num = test_result_boxes.shape[0]
frame_num = int(test_result_boxes[test_result_boxes_num - 1][0])

font = cv2.FONT_HERSHEY_SIMPLEX

for frame_idx in range(1, frame_num+1):
    curr_frame_path = data_set_root + data_set_name + '\\img1\\' + ('%06d' % frame_idx) + '.jpg'
    curr_result_frame_path = data_set_root + data_set_name + '\\result\\' + ('%06d' % frame_idx) + '.jpg'
    img = cv2.imread(curr_frame_path)
    curr_locations = get_test_locations(frame_idx, test_result_boxes)
    if curr_locations == None:
        continue
    for location in curr_locations:
        print location
        cv2.rectangle(img, (int(location[2]), int(location[3])), (int(location[2] + location[4]), int(location[3] + location[5])), (0,0,0))
        cv2.putText(img, str(int(location[1])), (int(location[2]), int(location[3])), font, 0.5, (255,255,255))
    cv2.imwrite(curr_result_frame_path, img)