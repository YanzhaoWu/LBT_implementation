# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 14:34:46 2016

@author: v-yanzwu
"""

import numpy as np
import os
import cv2

from optical_flow_generator import *
from img_preprocess import *

import h5py
def save_data_as_hdf5(hdf5_data_filename, data, label, location, time_stamp, scores):
    '''
    HDF5 is one of the data formats Caffe accepts
    '''
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data.astype(np.float32)
        f['label'] = label.astype(np.uint8)
        f['location'] = location.astype(np.uint8)
        f['timestamp'] = time_stamp.astype(np.uint8)
        f['score'] = scores.astype(np.float32)

def is_the_same(coordinate1, coordinate2, ground_truth):
    ground_truth_num = ground_truth.shape[0]
    if (coordinate1[0] == coordinate2[0]) and (coordinate1[2] == coordinate2[2]) \
        and (coordinate1[3] == coordinate2[3]) and (coordinate1[4] == coordinate2[4]) \
        and (coordinate1[5] == coordinate2[5]) and (coordinate1[6] == coordinate2[6]) :
        return True
    # Default parameters
    object_idx1 = 0
    object_idx2 = 1
    for idx in range(0, ground_truth_num):
        if (coordinate1[0] == ground_truth[idx][0]) and (coordinate1[2] == ground_truth[idx][2]) \
            and (coordinate1[3] == ground_truth[idx][3]) and (coordinate1[4] == ground_truth[idx][4]) \
            and (coordinate1[5] == ground_truth[idx][5]) :
            object_idx1 = ground_truth[idx][1]
        if (coordinate2[0] == ground_truth[idx][0]) and (coordinate2[2] == ground_truth[idx][2]) \
            and (coordinate2[3] == ground_truth[idx][3]) and (coordinate2[4] == ground_truth[idx][4]) \
            and (coordinate2[5] == ground_truth[idx][5]) :
            object_idx2 = ground_truth[idx][1]

    return object_idx1 == object_idx2

train_data_path = 'D:\\SharedData\\LBT_Train_Set\\2DMOT2015\\train\\'

train_data_set = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
train_data_set_num = len(train_data_set)

train_data_generate_num_ground_truth_positive = 1500
train_data_generate_num_ground_truth_negative = 750
train_data_generate_num_detection_negative = 750
train_data_generate_num = train_data_generate_num_ground_truth_positive +\
                          train_data_generate_num_ground_truth_negative +\
                          train_data_generate_num_detection_negative

                          

train_data_img_size = (121, 53, 10)

for train_data_set_id in range(0, train_data_set_num):
    
    train_data_img = np.zeros((train_data_generate_num, train_data_img_size[2], train_data_img_size[0], train_data_img_size[1]))
    train_data_location = np.zeros((train_data_generate_num, 2, 6))
    train_data_time_stamp = np.zeros((train_data_generate_num, 2))
    train_data_scores = np.zeros((train_data_generate_num, 2))
    train_data_labels = np.zeros(train_data_generate_num)
    
    ground_truth_boxes = np.loadtxt(train_data_path + train_data_set[train_data_set_id] + '\\gt\\gt.txt', delimiter = ',')
    ground_truth_boxes_num = ground_truth_boxes.shape[0]
    frame_num = int(ground_truth_boxes[ground_truth_boxes_num - 1][0])
    '''
    # Optical flow & Preprocess
    if not os.path.exists(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow'):
        os.mkdir(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow')
    if not os.path.exists(train_data_path + train_data_set[train_data_set_id] + '\\preprossed'):
        os.mkdir(train_data_path + train_data_set[train_data_set_id] + '\\preprossed')
    for frame_idx in range(1, frame_num):
        img1 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\img1\\' + ('%06d' % frame_idx) + '.jpg')
        img2 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\img1\\' + ('%06d' % (frame_idx + 1)) + '.jpg')
        optical_flow_img = generate_optical_flow(img1, img2, img1.shape)
        cv2.imwrite(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow\\' +  ('%06d' % frame_idx) + '.jpg', optical_flow_img)
        img1_preprocessed = img_preprocess_func(img1)
        cv2.imwrite(train_data_path + train_data_set[train_data_set_id] + '\\preprossed\\' + ('%06d' % frame_idx) + '.jpg', img1_preprocessed)
    '''
    ground_truth_positive = train_data_generate_num_ground_truth_positive
    ground_truth_negative = train_data_generate_num_ground_truth_negative
    counter = 0
    while (ground_truth_positive > 0) or (ground_truth_negative > 0):
        random_1 = np.random.randint(0, ground_truth_boxes_num)
        random_2 = np.random.randint(0, ground_truth_boxes_num)
        
        if (ground_truth_positive > 0) and (ground_truth_boxes[random_1][0] != ground_truth_boxes[random_2][0]) \
            and (ground_truth_boxes[random_1][1] == ground_truth_boxes[random_2][1]) \
            and (ground_truth_boxes[random_1][0] - ground_truth_boxes[random_2][0] < 15) \
            and (ground_truth_boxes[random_2][0] - ground_truth_boxes[random_2][1] < 15) \
            and (ground_truth_boxes[random_1][0] < frame_num) \
            and (ground_truth_boxes[random_2][0] < frame_num):
            # Positive examples
            img1 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\preprossed\\' + ('%06d' % ground_truth_boxes[random_1][0]) + '.jpg')
            img2 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\preprossed\\' + ('%06d' % ground_truth_boxes[random_2][0]) + '.jpg')
            optical_flow1 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow\\' + ('%06d' % ground_truth_boxes[random_1][0]) + '.jpg')
            optical_flow2 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow\\' + ('%06d' % ground_truth_boxes[random_2][0]) + '.jpg')
            train_data_time_stamp[counter][0] = ground_truth_boxes[random_1][0]
            train_data_time_stamp[counter][1] = ground_truth_boxes[random_2][0]
            train_data_scores[counter][0] = 100
            train_data_scores[counter][1] = 100
            train_data_location[counter][0] = ground_truth_boxes[random_1][0:6]
            train_data_location[counter][1] = ground_truth_boxes[random_2][0:6]
            tmp_img = np.zeros((train_data_img_size[2], train_data_img_size[0], train_data_img_size[1]))
            tmp_img1 = img1[int(max(0, ground_truth_boxes[random_1][3])) : int(max(0, ground_truth_boxes[random_1][3]) + max(0, ground_truth_boxes[random_1][5] - 1)), int(max(0, ground_truth_boxes[random_1][2])) : int(max(0, ground_truth_boxes[random_1][2]) + max(0, ground_truth_boxes[random_1][4] - 1))]
            tmp_img2 = img2[int(max(0, ground_truth_boxes[random_2][3])) : int(max(0, ground_truth_boxes[random_2][3]) + max(0, ground_truth_boxes[random_2][5] - 1)), int(max(0, ground_truth_boxes[random_2][2])) : int(max(0, ground_truth_boxes[random_2][2]) + max(0, ground_truth_boxes[random_2][4] - 1))]
            tmp_optical_flow1 = optical_flow1[int(max(0, ground_truth_boxes[random_1][3])) : int(max(0, ground_truth_boxes[random_1][3]) + max(0, ground_truth_boxes[random_1][5] - 1)), int(max(0, ground_truth_boxes[random_1][2])) : int(max(0, ground_truth_boxes[random_1][2]) + max(0, ground_truth_boxes[random_1][4] - 1))]
            tmp_optical_flow2 = optical_flow2[int(max(0, ground_truth_boxes[random_2][3])) : int(max(0, ground_truth_boxes[random_2][3]) + max(0, ground_truth_boxes[random_2][5] - 1)), int(max(0, ground_truth_boxes[random_2][2])) : int(max(0, ground_truth_boxes[random_2][2]) + max(0, ground_truth_boxes[random_2][4] - 1))]
            tmp_img1 = cv2.resize(tmp_img1, (train_data_img_size[1], train_data_img_size[0]))
            tmp_img2 = cv2.resize(tmp_img2, (train_data_img_size[1], train_data_img_size[0]))
            tmp_optical_flow1 = cv2.resize(tmp_optical_flow1, (train_data_img_size[1], train_data_img_size[0]))
            tmp_optical_flow2 = cv2.resize(tmp_optical_flow2, (train_data_img_size[1], train_data_img_size[0]))
            for idx in range(0, tmp_img1.shape[2]):
                tmp_img[idx, ...] = tmp_img1[..., idx]
                #print idx, idx
            for idx in range(tmp_img1.shape[2], tmp_img1.shape[2] + tmp_optical_flow1.shape[2] - 1):
                tmp_img[idx, ...] = tmp_optical_flow1[..., idx - tmp_img1.shape[2] + 1]
                #print idx, idx - tmp_img1.shape[2] + 1
            for idx in range(tmp_img1.shape[2] + tmp_optical_flow1.shape[2] - 1, tmp_img1.shape[2] + tmp_optical_flow1.shape[2] + tmp_img2.shape[2] - 1):
                tmp_img[idx, ...] = tmp_img2[..., idx - tmp_img1.shape[2] - tmp_img1.shape[2] + 1]
                #print idx, idx - tmp_img1.shape[2] - tmp_optical_flow1.shape[2] + 1
            for idx in range(tmp_img1.shape[2] + tmp_optical_flow1.shape[2] + tmp_img2.shape[2] - 1, tmp_img1.shape[2] + tmp_optical_flow1.shape[2] + tmp_img2.shape[2] + tmp_optical_flow2.shape[2] - 2):
                tmp_img[idx, ...] = tmp_optical_flow2[..., idx - tmp_img1.shape[2] - tmp_optical_flow1.shape[2] - tmp_img2.shape[2] + 2]
                #print idx, idx - tmp_img1.shape[2] - tmp_optical_flow1.shape[2] - tmp_img2.shape[2] + 2
            train_data_img[counter, ...] = tmp_img
            train_data_labels[counter] = 1
            counter += 1
            #print 'ground_truth_positive'
            #print counter
            ground_truth_positive -= 1
        elif (ground_truth_negative > 0) and (ground_truth_boxes[random_1][1] != ground_truth_boxes[random_2][1]) \
            and (ground_truth_boxes[random_1][0] != ground_truth_boxes[random_2][0]) \
            and (ground_truth_boxes[random_1][0] < frame_num) \
            and (ground_truth_boxes[random_2][0] < frame_num) :
            # Negative examples
            img1 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\preprossed\\' + ('%06d' % ground_truth_boxes[random_1][0]) + '.jpg')
            img2 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\preprossed\\' + ('%06d' % ground_truth_boxes[random_2][0]) + '.jpg')
            optical_flow1 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow\\' + ('%06d' % ground_truth_boxes[random_1][0]) + '.jpg')
            optical_flow2 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow\\' + ('%06d' % ground_truth_boxes[random_2][0]) + '.jpg')
            train_data_time_stamp[counter][0] = ground_truth_boxes[random_1][0]
            train_data_time_stamp[counter][1] = ground_truth_boxes[random_2][0]
            train_data_scores[counter][0] = 100
            train_data_scores[counter][1] = 100
            train_data_location[counter][0] = ground_truth_boxes[random_1][0:6]
            train_data_location[counter][1] = ground_truth_boxes[random_2][0:6]
            tmp_img = np.zeros((train_data_img_size[2], train_data_img_size[0], train_data_img_size[1]))
            tmp_img1 = img1[int(max(0, ground_truth_boxes[random_1][3])): int(max(0, ground_truth_boxes[random_1][3]) + max(0, ground_truth_boxes[random_1][5] - 1)), int(max(0, ground_truth_boxes[random_1][2])) : int(max(0, ground_truth_boxes[random_1][2]) + max(0, ground_truth_boxes[random_1][4] - 1))]
            tmp_img2 = img2[int(max(0, ground_truth_boxes[random_2][3])): int(max(0, ground_truth_boxes[random_2][3]) + max(0, ground_truth_boxes[random_2][5] - 1)), int(max(0, ground_truth_boxes[random_2][2])) : int(max(0, ground_truth_boxes[random_2][2]) + max(0, ground_truth_boxes[random_2][4] - 1))]
            tmp_optical_flow1 = optical_flow1[int(max(0, ground_truth_boxes[random_1][3])) : int(max(0, ground_truth_boxes[random_1][3]) + max(0, ground_truth_boxes[random_1][5] - 1)), int(max(0, ground_truth_boxes[random_1][2])) : int(max(0, ground_truth_boxes[random_1][2]) + max(0, ground_truth_boxes[random_1][4] - 1))]
            tmp_optical_flow2 = optical_flow2[int(max(0, ground_truth_boxes[random_2][3])) : int(max(0, ground_truth_boxes[random_2][3]) + max(0, ground_truth_boxes[random_2][5] - 1)), int(max(0, ground_truth_boxes[random_2][2])) : int(max(0, ground_truth_boxes[random_2][2]) + max(0, ground_truth_boxes[random_2][4] - 1))]
            tmp_img1 = cv2.resize(tmp_img1, (train_data_img_size[1], train_data_img_size[0]))
            tmp_img2 = cv2.resize(tmp_img2, (train_data_img_size[1], train_data_img_size[0]))
            tmp_optical_flow1 = cv2.resize(tmp_optical_flow1, (train_data_img_size[1], train_data_img_size[0]))
            tmp_optical_flow2 = cv2.resize(tmp_optical_flow2, (train_data_img_size[1], train_data_img_size[0]))
            for idx in range(0, tmp_img1.shape[2]):
                tmp_img[idx, ...] = tmp_img1[..., idx]
                #print idx, idx
            for idx in range(tmp_img1.shape[2], tmp_img1.shape[2] + tmp_optical_flow1.shape[2] - 1):
                tmp_img[idx, ...] = tmp_optical_flow1[..., idx - tmp_img1.shape[2] + 1]
                #print idx, idx - tmp_img1.shape[2] + 1
            for idx in range(tmp_img1.shape[2] + tmp_optical_flow1.shape[2] - 1, tmp_img1.shape[2] + tmp_optical_flow1.shape[2] + tmp_img2.shape[2] - 1):
                tmp_img[idx, ...] = tmp_img2[..., idx - tmp_img1.shape[2] - tmp_optical_flow1.shape[2] + 1]
                #print idx, idx - tmp_img1.shape[2] - tmp_optical_flow1.shape[2] + 1
            for idx in range(tmp_img1.shape[2] + tmp_optical_flow1.shape[2] + tmp_img2.shape[2] - 1, tmp_img1.shape[2] + tmp_optical_flow1.shape[2] + tmp_img2.shape[2] + tmp_optical_flow2.shape[2] - 2):
                tmp_img[idx, ...] = tmp_optical_flow2[..., idx - tmp_img1.shape[2] - tmp_optical_flow1.shape[2] - tmp_img2.shape[2] + 2]
                #print idx, idx - tmp_img1.shape[2] - tmp_optical_flow1.shape[2] - tmp_img2.shape[2] + 2
            train_data_img[counter, ...] = tmp_img
            train_data_labels[counter] = 0
            counter += 1
            #print 'ground_truth_negative'
            #print counter
            ground_truth_negative -= 1
    

    
    detection_boxes = np.loadtxt(train_data_path + train_data_set[train_data_set_id] + '\\det\\det.txt', delimiter = ',')
    detection_boxes_num = detection_boxes.shape[0]
    detection_negative = train_data_generate_num_detection_negative
    
    while (detection_negative > 0):
        random_1 = np.random.randint(0, detection_boxes_num)
        random_2 = np.random.randint(0, detection_boxes_num)
        if is_the_same(detection_boxes[random_1], detection_boxes[random_2], ground_truth_boxes) \
            or (detection_boxes[random_1][0] == detection_boxes[random_2][0]) \
            or (detection_boxes[random_1][0] >= frame_num) \
            or (detection_boxes[random_2][0] >= frame_num) :
            continue
        img1 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\preprossed\\' + ('%06d' % detection_boxes[random_1][0]) + '.jpg')
        img2 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\preprossed\\' + ('%06d' % detection_boxes[random_2][0]) + '.jpg')
        optical_flow1 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow\\' + ('%06d' % detection_boxes[random_1][0]) + '.jpg')
        optical_flow2 = cv2.imread(train_data_path + train_data_set[train_data_set_id] + '\\optical_flow\\' + ('%06d' % detection_boxes[random_2][0]) + '.jpg')
        train_data_time_stamp[counter][0] = detection_boxes[random_1][0]
        train_data_time_stamp[counter][1] = detection_boxes[random_2][0]
        train_data_scores[counter][0] = detection_boxes[random_1][6]
        train_data_scores[counter][1] = detection_boxes[random_2][6]
        train_data_location[counter][0] = detection_boxes[random_1][0:6]
        train_data_location[counter][1] = detection_boxes[random_2][0:6]
        tmp_img = np.zeros((train_data_img_size[2], train_data_img_size[0], train_data_img_size[1]))
        tmp_img1 = img1[int(max(0, detection_boxes[random_1][3])) : int(max(0, detection_boxes[random_1][3]) + max(0, detection_boxes[random_1][5] - 1)), int(max(0, detection_boxes[random_1][2])) : int(max(0, detection_boxes[random_1][2]) + max(0, detection_boxes[random_1][4] - 1))]
        tmp_img2 = img2[int(max(0, detection_boxes[random_2][3])) : int(max(0, detection_boxes[random_2][3]) + max(0, detection_boxes[random_2][5] - 1)), int(max(0, detection_boxes[random_2][2])) : int(max(0, detection_boxes[random_2][2]) + max(0, detection_boxes[random_2][4] - 1))]
        tmp_optical_flow1 = optical_flow1[int(max(0, detection_boxes[random_1][3])) : int(max(0, detection_boxes[random_1][3]) + max(0, detection_boxes[random_1][5] - 1)), int(max(0, detection_boxes[random_1][2])) : int(max(0, detection_boxes[random_1][2]) + max(0, detection_boxes[random_1][4] - 1))]
        tmp_optical_flow2 = optical_flow2[int(max(0, detection_boxes[random_2][3])) : int(max(0, detection_boxes[random_2][3]) + max(0, detection_boxes[random_2][5] - 1)), int(max(0, detection_boxes[random_2][2])) : int(max(0, detection_boxes[random_2][2]) + max(0, detection_boxes[random_2][4] - 1))]
        tmp_img1 = cv2.resize(tmp_img1, (train_data_img_size[1], train_data_img_size[0]))
        tmp_img2 = cv2.resize(tmp_img2, (train_data_img_size[1], train_data_img_size[0]))
        tmp_optical_flow1 = cv2.resize(tmp_optical_flow1, (train_data_img_size[1], train_data_img_size[0]))
        tmp_optical_flow2 = cv2.resize(tmp_optical_flow2, (train_data_img_size[1], train_data_img_size[0]))
        for idx in range(0, tmp_img1.shape[2]):
            tmp_img[idx, ...] = tmp_img1[..., idx]
            #print idx, idx
        for idx in range(tmp_img1.shape[2], tmp_img1.shape[2] + tmp_optical_flow1.shape[2] - 1):
            tmp_img[idx, ...] = tmp_optical_flow1[..., idx - tmp_img1.shape[2] + 1]
            #print idx, idx - tmp_img1.shape[2] + 1
        for idx in range(tmp_img1.shape[2] + tmp_optical_flow1.shape[2] - 1, tmp_img1.shape[2] + tmp_optical_flow1.shape[2] + tmp_img2.shape[2] - 1):
            tmp_img[idx, ...] = tmp_img2[..., idx - tmp_img1.shape[2] - tmp_optical_flow1.shape[2] + 1]
            #print idx, idx - tmp_img1.shape[2] - tmp_optical_flow1.shape[2] + 1
        for idx in range(tmp_img1.shape[2] + tmp_optical_flow1.shape[2] + tmp_img2.shape[2] - 1, tmp_img1.shape[2] + tmp_optical_flow1.shape[2] + tmp_img2.shape[2] + tmp_optical_flow2.shape[2] - 2):
            tmp_img[idx, ...] = tmp_optical_flow2[..., idx - tmp_img1.shape[2] - tmp_optical_flow1.shape[2] - tmp_img2.shape[2] + 2]
            #print idx, idx - tmp_img1.shape[2] - tmp_optical_flow1.shape[2] - tmp_img2.shape[2] + 2
        train_data_img[counter, ...] = tmp_img
        train_data_labels[counter] = 0
        counter += 1
        #print 'negative'
        #print counter
        detection_negative -= 1
    
    shuffle_idx = np.random.permutation(train_data_labels.shape[0])
    train_data_img = train_data_img[shuffle_idx]
    train_data_labels = train_data_labels[shuffle_idx]
    train_data_location = train_data_location[shuffle_idx]
    train_data_time_stamp = train_data_time_stamp[shuffle_idx]
    train_data_scores = train_data_scores[shuffle_idx]

    save_data_as_hdf5(train_data_set[train_data_set_id] + '.hdf5', train_data_img, train_data_labels, train_data_location, train_data_time_stamp, train_data_scores)