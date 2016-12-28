# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:31:33 2016

@author: v-yanzwu
"""

import h5py
import numpy as np

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

train_data_set = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
train_data_img_size = (121, 53, 10)
train_data_generate_num_ground_truth_positive = 1000
train_data_generate_num_ground_truth_negative = 1000
train_data_generate_num_detection_negative = 1000
train_data_generate_num = train_data_generate_num_ground_truth_positive +\
                          train_data_generate_num_ground_truth_negative +\
                          train_data_generate_num_detection_negative

train_data_set_num = len(train_data_set)
data_num = train_data_generate_num * train_data_set_num
data = np.zeros((data_num, train_data_img_size[2], train_data_img_size[0], train_data_img_size[1]))
label = np.zeros(data_num)
location = np.zeros((data_num, 2, 6))
timestamp = np.zeros((data_num, 2))
score = np.zeros((data_num, 2))

counter = 0
for train_data_set_idx in range(0, train_data_set_num):
    hdf5_file = h5py.File(train_data_set[train_data_set_idx] + '.hdf5', 'r')
    data[counter : counter + train_data_generate_num, ...] = hdf5_file['data'][...]
    label[counter : counter + train_data_generate_num, ...] = hdf5_file['label'][...]
    location[counter : counter + train_data_generate_num, ...] = hdf5_file['location'][...]
    timestamp[counter : counter + train_data_generate_num, ...] = hdf5_file['timestamp'][...]
    score[counter : counter + train_data_generate_num, ...] = hdf5_file['score'][...]
    
    counter += train_data_generate_num
    
save_data_as_hdf5('full_train_data.hdf5', data, label, location, timestamp, score)