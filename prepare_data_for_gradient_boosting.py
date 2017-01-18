# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 19:22:44 2017

@author: v-yanzwu
"""
import caffe

import h5py
import numpy as np
caffe.set_mode_cpu()
# Gradient Boost

hdf5_data_path = 'full_train_data.hdf5'
hdf5_file = h5py.File(hdf5_data_path, 'r')

labels = hdf5_file['label'][:]
locations = hdf5_file['location'][:]
times = hdf5_file['timestamp'][:]
train_data_img = hdf5_file['data'][:]

net = caffe.Net('LBT_deploy.prototxt', 'lbt__iter_45000.caffemodel', caffe.TEST)
train_samples = np.zeros((labels.shape[0], 516))
for i in range(0, labels.shape[0]):
    
    x0 = np.array([max(0, locations[i][0][2]) + 0.5 * max(0, locations[i][0][4]), max(0, locations[i][0][3]) + 0.5 * max(0, locations[i][0][5])]) # Center Location
    x1 = np.array([max(0, locations[i][1][2]) + 0.5 * max(0, locations[i][1][4]), max(0, locations[i][1][3]) + 0.5 * max(0, locations[i][1][5])]) # Center Location
    s0 = np.array([max(0, locations[i][0][4]), max(0, locations[i][0][5])])
    s1 = np.array([max(0, locations[i][1][4]), max(0, locations[i][1][5])])
    t0 = times[i][0]
    t1 = times[i][1]
    #if t0 == t1:
    #    continue
    relative_size_change = (s0 - s1) / (s0 + s1)
    relative_velocity = (x0 - x1) / (t1 - t0)
    train_samples[i][0:2] = relative_size_change
    train_samples[i][2:4] = relative_velocity
    net.blobs['data'].data[...] = train_data_img[i]
    tmp_out = net.forward()
    cnn_prediction = tmp_out['fc6'].copy()
    train_samples[i][4:516] = cnn_prediction

train_samples[np.isfinite(train_samples) == False] = 10000
train_samples[np.isnan(train_samples) == True] = 0  

def save_data_as_hdf5_prediction(hdf5_data_filename, data, label, location, time_stamp, cnn_prediction):
    # HDF5 is one of the data formats Caffe accepts
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data.astype(np.float32)
        f['label'] = label.astype(np.float32)
        f['location'] = location.astype(np.uint8)
        f['timestamp'] = time_stamp.astype(np.uint8)
        f['cnnprediction'] = cnn_prediction.astype(np.float32)

save_data_as_hdf5_prediction('full_train_data_with_cnn_predicts_512.hdf5', train_data_img, labels, locations, times, train_samples)