# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:10:55 2016

@author: v-yanzwu
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.datasets import load_digits
from stacked_generalizer import StackedGeneralizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import caffe


import numpy as np
import os
import cv2

from optical_flow_generator import *
from img_preprocess import *

caffe.set_mode_cpu()
# Gradient Boost

hdf5_data_path = 'full_train_data.hdf5'
hdf5_file = h5py.File(hdf5_data_path, 'r')

labels = hdf5_file['label'][:]
locations = hdf5_file['location'][:]
times = hdf5_file['timestamp'][:]
train_data_img = hdf5_file['data'][:]

net = caffe.Net('LBT_deploy.prototxt', 'lbt__iter_500.caffemodel', caffe.TEST)
train_samples = np.zeros((labels.shape[0], 516))
for i in range(0, labels.shape[0]):
    
    x0 = np.array([max(0, locations[i][0][2]) + 0.5 * max(0, locations[i][0][4]), max(0, locations[i][0][3]) + 0.5 * max(0, locations[i][0][5])]) # Center Location
    x1 = np.array([max(0, locations[i][1][2]) + 0.5 * max(0, locations[i][2][4]), max(0, locations[i][1][3]) + 0.5 * max(0, locations[i][1][5])]) # Center Location
    s0 = np.array([max(0, locations[i][0][4]), max(0, locations[i][0][5])])
    s1 = np.array([max(0, locations[i][1][4]), max(0, locations[i][1][5])])
    t0 = times[i][0]
    t1 = times[i][1]
    
    relative_size_change = (s0 - s1) / (s0 + s1)
    relative_velocity = (x0 - x1) / (t1 - t0)
    train_samples[i][0:2] = relative_size_change
    train_samples[i][2:4] = relative_velocity
    net.blobs['data'].data[...] = train_data_img[i]
    tmp_out = net.forward()
    cnn_prediction = tmp_out['fc6'].copy()
    train_samples[i][4:516] = cnn_prediction
    
'''
def save_data_as_hdf5_prediction(hdf5_data_filename, label, location, time_stamp, cnn_prediction):
    # HDF5 is one of the data formats Caffe accepts
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data.astype(np.float32)
        f['label'] = label.astype(np.float32)
        f['location'] = location.astype(np.uint8)
        f['timestamp'] = time_stamp.astype(np.uint8)
        f['cnnprediction'] = cnn_prediction.astype(np.float32)

save_data_as_hdf5_prediction('full_train_data_with_predictions.hdf5', labels, locations, times, train_samples[:, 0:516])
'''

VERBOSE = True
N_FOLDS = 5

# load data and shuffle observations
# data = load_digits()

#X = data.data # Change to another type
#y = data.target
#print relative_size_change.shape
#print relative_velocity.shape
#print cnn_prediction.shape
#
##test_sample = [relative_size_change, relative_velocity, cnn_prediction]
##train_sample = np.array(test_sample)
#
#test_sample = np.append(np.append(relative_size_change, relative_velocity), cnn_prediction)
#print test_sample.shape

#train_sample = np.random.randint(0, 255,(100,test_sample.shape[0]))

# print test_sample.shape
# print train_sample

#X = np.random.randint(0,255, test_sample.shape)
#y = np.random.randint(0,2, train_sample.shape[0])

#print y.shape
#print y

#shuffle_idx = np.random.permutation(labels.shape[0])

#train_samples = train_samples[shuffle_idx]
#labels = labels[shuffle_idx]
#train_data_img = train_data_img[shuffle_idx]
#locations = locations[shuffle_idx]
#times = times[shuffle_idx]

# hold out 20 percent of data for testing accuracy
#train_prct = 1
#n_train = int(round(train_samples.shape[0] * train_prct))

# define base models
base_models = [GradientBoostingClassifier(n_estimators=100),
               GradientBoostingClassifier(n_estimators=100),
               GradientBoostingClassifier(n_estimators=100),
               GradientBoostingClassifier(n_estimators=100),
               GradientBoostingClassifier(n_estimators=100)]

# define blending model
blending_model = LogisticRegression()

# initialize multi-stage model
sg = StackedGeneralizer(base_models, blending_model, 
                        n_folds=N_FOLDS, verbose=VERBOSE)

# fit model
sg.fit(train_samples, labels)

# Generate Test Data

test_data_path = 'D:\\SharedData\\LBT_Train_Set\\2DMOT2015\\test\\'

#test_data_set = ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher', 'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1']

test_data_set = ['KITTI-16']
for 

test_data_locations = 
test_times = times[n_train:]
test_data_img = train_data_img[n_train:]

n_test = test_locations.shape[0] * 2

isolated_test_locations = np.zeros((n_test, test_locations.shape[2]))
isolated_test_times = np.zeros(n_test)
isolated_test_data_img = np.zeros((n_test, 5, train_data_img.shape[2], train_data_img.shape[3]))

for i in range(0, n_test, 2):
    isolated_test_data_img[i, ...] = test_data_img[i / 2, 0:5, ...]
    isolated_test_data_img[i + 1, ...] = test_data_img[i / 2, 5:10, ...]
    isolated_test_locations[i, ...] = test_locations[i / 2, 0, ...]
    isolated_test_locations[i + 1, ...] = test_locations[i / 2, 1, ...]
    isolated_test_times[i] = test_times[i / 2, 0]
    isolated_test_times[i + 1] = test_times[i / 2, 1]

candidate_samples = np.zeros((n_test * n_test, 516), dtype = np.float32)

for i in range(0, n_test):
    for j in range(0, n_test):
        x0 = np.array([isolated_test_locations[i][0], isolated_test_locations[i][2]])
        x1 = np.array([isolated_test_locations[j][0], isolated_test_locations[j][2]])
        s0 = np.array([isolated_test_locations[i][2] - isolated_test_locations[i][0], isolated_test_locations[i][3] - isolated_test_locations[i][1]])
        s1 = np.array([isolated_test_locations[j][2] - isolated_test_locations[j][0], isolated_test_locations[j][3] - isolated_test_locations[j][1]])
        t0 = times[i]
        t1 = times[j]
    
        relative_size_change = (s0 - s1) / (s0 + s1)
        relative_velocity = (x0 - x1) / (t1 - t0)
        candidate_samples[i * n_test + j][0:2] = relative_size_change
        candidate_samples[i * n_test + j][2:4] = relative_velocity
        net.blobs['data'].data[0, 0:5, ...] = isolated_test_data_img[i]
        net.blobs['data'].data[0, 5:10, ...] = isolated_test_data_img[j]
        tmp_out = net.forward()
        cnn_prediction = tmp_out['fc6'].copy()
        print cnn_prediction
        candidate_samples[i * n_test + j][4:516] = cnn_prediction

candidate_samples[np.isfinite(candidate_samples) == False] = 0
candidate_samples[np.isnan(candidate_samples) == True] = 0
pred = sg.predict(candidate_samples)
pred_labels = np.zeros(pred.shape[0])
for i in range(0, pred.shape[0]):
    if pred[i, 0] > pred[i, 1]:
        pred_labels[i] = pred[i, 0]
    else:
        pred_labels[i] = pred[i, 1]
'''
def save_data_as_hdf5_results(hdf5_data_filename, data, predict_labels):
    #HDF5 is one of the data formats Caffe accepts
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data.astype(np.float32)
        f['label'] = predict_labels.astype(np.float32)

save_data_as_hdf5_results('test_data_full_with_prediction_.hdf5', isolated_test_data_img[:, 0:3, ...], pred_labels) 
'''


# test accuracy
pred = sg.predict(train_samples[n_train:])
pred_classes = [np.argmax(p) for p in pred]
#
_ = sg.evaluate(labels[n_train:], pred_classes)