# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:41:22 2016

@author: v-yanzwu
"""
import numpy as np

import caffe
import h5py

caffe.set_mode_cpu()

hdf5_data_path = 'D:\\Projects\\LBT_implementation\\test_data.hdf5' #\\train_data_backup_correct\\small_ones\\ADL-Rundle-6.hdf5'
hdf5_file = h5py.File(hdf5_data_path, 'r')

labels = hdf5_file['label'][:]

train_data_img = hdf5_file['data'][:]

net = caffe.Net('LBT_deploy_test.prototxt', 'lbt__iter_45000.caffemodel', caffe.TEST)
train_samples = np.zeros((labels.shape[0], 512))
for i in range(0, labels.shape[0]):
    net.blobs['data'].data[0, ...] = train_data_img[i]
    net.blobs['label'].data[...] = labels[i]
    tmp_out = net.forward()
    print tmp_out
    #cnn_prediction = tmp_out['fc6'].copy()
    #train_samples[i][0:512] = cnn_prediction.squeeze()