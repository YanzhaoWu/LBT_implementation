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


# LinearProgramming
from pulp import LpProblem, LpVariable, LpInteger, LpMinimize, lpSum, LpStatus
import networkx as nx

def c_det(s_i, v_det, v_link):
# input : s_i should be normalized to {0,1}
    if (s_i < v_det):
        return  (1 + (-s_i) / v_det)
    else:
        return (-1 + (-s_i + 1) / (1 - v_link))
        
        
def c_t(s_i_j, v_link):
    #print s_i_j
    if (s_i_j < v_link):
        return (1 + (-s_i_j) / v_link)
    else:
        return (-1 + (-s_i_j + 1) / (1 - v_link))

# Todo c_in = c_out, v_det and v_link
c_in = 0.5
c_out = 0.5

v_det = 0.35
v_link = 0.35




caffe.set_mode_cpu()
# Gradient Boost

hdf5_data_path = 'D:\\Projects\\LBT_implementation\\train_data_backup_correct\\small_ones\\ADL-Rundle-6.hdf5'
hdf5_file = h5py.File(hdf5_data_path, 'r')

labels = hdf5_file['label'][:]
locations = hdf5_file['location'][:]
times = hdf5_file['timestamp'][:]
train_data_img = hdf5_file['data'][:]

net = caffe.Net('LBT_deploy.prototxt', 'lbt__iter_500.caffemodel', caffe.TEST)
train_samples = np.zeros((labels.shape[0], 516))
for i in range(0, labels.shape[0]):
    
    x0 = np.array([max(0, locations[i][0][2]) + 0.5 * max(0, locations[i][0][4]), max(0, locations[i][0][3]) + 0.5 * max(0, locations[i][0][5])]) # Center Location
    x1 = np.array([max(0, locations[i][1][2]) + 0.5 * max(0, locations[i][1][4]), max(0, locations[i][1][3]) + 0.5 * max(0, locations[i][1][5])]) # Center Location
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

def get_detection_locations_and_scores(frame_idx, detection_boxes):
    i = 0
    is_first_time = True
    result = None
    while (detection_boxes[i][0] <= frame_idx):
        if (detection_boxes[i][0] == frame_idx):
            if is_first_time:
                result = np.array(detection_boxes[i][0:7])
                #print result
                is_first_time = False
            else:
                result = np.vstack((result, detection_boxes[i][0:7]))
                #print result
        i += 1
    return result            


test_data_path = 'D:\\SharedData\\LBT_Train_Set\\2DMOT2015\\test\\'
test_data_img_size = (121, 53, 10)
#test_data_set = ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher', 'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1']

test_data_set = ['KITTI-16']
test_data_set_num = len(test_data_set)

for test_data_set_idx in range(0, test_data_set_num):
    detection_boxes = np.loadtxt(test_data_path + test_data_set[test_data_set_idx] + '\\det\\det.txt', delimiter = ',')
    detection_boxes_num = detection_boxes.shape[0]
    frame_num = int(detection_boxes[detection_boxes_num - 1][0])
    is_first_result = True
    result = None
    for frame_idx in range(1, frame_num):
        frame_1_locations_and_scores = get_detection_locations_and_scores(frame_idx, detection_boxes)
        frame_2_locations_and_scores = get_detection_locations_and_scores(frame_idx + 1, detection_boxes)
        
        n_test = frame_1_locations_and_scores.shape[0] + frame_2_locations_and_scores.shape[0]
        
        img1 = cv2.imread(test_data_path + test_data_set[test_data_set_idx] + '\\preprossed\\' + ('%06d' % frame_idx) + '.jpg')
        img2 = cv2.imread(test_data_path + test_data_set[test_data_set_idx] + '\\preprossed\\' + ('%06d' % (frame_idx + 1)) + '.jpg')
        optical_flow1 = cv2.imread(test_data_path + test_data_set[test_data_set_idx] + '\\optical_flow\\' + ('%06d' % frame_idx) + '.jpg')
        optical_flow2 = cv2.imread(test_data_path + test_data_set[test_data_set_idx] + '\\optical_flow\\' + ('%06d' % (frame_idx + 1)) + '.jpg')
        
        isolated_test_locations = np.zeros((n_test, 6))
        isolated_test_times = np.zeros(n_test)
        isolated_test_scores = np.zeros(n_test)
        isolated_test_data_img = np.zeros((n_test, 5, train_data_img.shape[2], train_data_img.shape[3]))
        
        counter = 0
        for idx in range(0, frame_1_locations_and_scores.shape[0]):
            isolated_test_times[counter] = frame_1_locations_and_scores[idx][0]
            isolated_test_locations[counter] = frame_1_locations_and_scores[idx][0:6]
            isolated_test_scores[counter] = frame_1_locations_and_scores[idx][6]
            
            tmp_img = img1[int(max(0, frame_1_locations_and_scores[idx][3])) : int(max(0, frame_1_locations_and_scores[idx][3]) + max(0, frame_1_locations_and_scores[idx][5] - 1)), int(max(0, frame_1_locations_and_scores[idx][2])) : int(max(0, frame_1_locations_and_scores[idx][2]) + max(0, frame_1_locations_and_scores[idx][4] - 1))]
            tmp_optical_flow = optical_flow1[int(max(0, frame_1_locations_and_scores[idx][3])) : int(max(0, frame_1_locations_and_scores[idx][3]) + max(0, frame_1_locations_and_scores[idx][5] - 1)), int(max(0, frame_1_locations_and_scores[idx][2])) : int(max(0, frame_1_locations_and_scores[idx][2]) + max(0, frame_1_locations_and_scores[idx][4] - 1))]
            for idx in range(0, tmp_img.shape[2]):
                isolated_test_data_img[counter, idx, ...] = cv2.resize(tmp_img[..., idx], (test_data_img_size[1], test_data_img_size[0]))
            for idx in range(tmp_img.shape[2], tmp_img.shape[2] + tmp_optical_flow.shape[2] - 1):
                isolated_test_data_img[counter, idx, ...] = cv2.resize(tmp_optical_flow[..., idx - tmp_img.shape[2] + 1], (test_data_img_size[1], test_data_img_size[0]))
            counter += 1
            
        for idx in range(0, frame_2_locations_and_scores.shape[0]):
            isolated_test_times[counter] = frame_2_locations_and_scores[idx][0]
            isolated_test_locations[counter] = frame_2_locations_and_scores[idx][0:6]
            isolated_test_scores[counter] = frame_2_locations_and_scores[idx][6]
            
            tmp_img = img2[int(max(0, frame_2_locations_and_scores[idx][3])) : int(max(0, frame_2_locations_and_scores[idx][3]) + max(0, frame_2_locations_and_scores[idx][5] - 1)), int(max(0, frame_2_locations_and_scores[idx][2])) : int(max(0, frame_2_locations_and_scores[idx][2]) + max(0, frame_2_locations_and_scores[idx][4] - 1))]
            tmp_optical_flow = optical_flow2[int(max(0, frame_2_locations_and_scores[idx][3])) : int(max(0, frame_2_locations_and_scores[idx][3]) + max(0, frame_2_locations_and_scores[idx][5] - 1)), int(max(0, frame_2_locations_and_scores[idx][2])) : int(max(0, frame_2_locations_and_scores[idx][2]) + max(0, frame_2_locations_and_scores[idx][4] - 1))]
            for idx in range(0, tmp_img.shape[2]):
                isolated_test_data_img[counter, idx, ...] = cv2.resize(tmp_img[..., idx], (test_data_img_size[1], test_data_img_size[0]))
            for idx in range(tmp_img.shape[2], tmp_img.shape[2] + tmp_optical_flow.shape[2] - 1):
                isolated_test_data_img[counter, idx, ...] = cv2.resize(tmp_optical_flow[..., idx - tmp_img.shape[2] + 1], (test_data_img_size[1], test_data_img_size[0]))
            counter += 1
        candidate_samples = np.zeros((n_test * n_test, 516), dtype = np.float32)
        
        for i in range(0, n_test):
            x0 = np.array([max(0, isolated_test_locations[i][2]) + 0.5 * max(0, isolated_test_locations[i][4]), max(0, isolated_test_locations[i][3]) + 0.5 * max(0, isolated_test_locations[i][5])]) # Center Location
            s0 = np.array([max(0, isolated_test_locations[i][4]), max(0, isolated_test_locations[i][5])])
            t0 = times[i]
            for j in range(0, n_test):
                x1 = np.array([max(0, isolated_test_locations[j][2]) + 0.5 * max(0, isolated_test_locations[j][4]), max(0, isolated_test_locations[j][3]) + 0.5 * max(0, isolated_test_locations[j][5])]) # Center Location
                s1 = np.array([max(0, isolated_test_locations[j][4]), max(0, isolated_test_locations[j][5])])    
                t1 = times[j]
            
                relative_size_change = (s0 - s1) / (s0 + s1)
                relative_velocity = (x0 - x1) / (t1 - t0)
                candidate_samples[i * n_test + j][0:2] = relative_size_change
                candidate_samples[i * n_test + j][2:4] = relative_velocity
                net.blobs['data'].data[0, 0:5, ...] = isolated_test_data_img[i]
                net.blobs['data'].data[0, 5:10, ...] = isolated_test_data_img[j]
                tmp_out = net.forward()
                cnn_prediction = tmp_out['fc6'].copy()
                # print cnn_prediction
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
        s_array = isolated_test_scores
        #print s_array
        s_matrix = np.array(pred_labels).reshape((n_test,n_test))
        #print s_matrix
        

        index = ' '

        for i in range(0, n_test):
            index += str(i)
            index += ' '

        index = index.split()

        prob = LpProblem('Binary Linear Programming', LpMinimize)
        fin = LpVariable.dicts('fin', index, lowBound=0, upBound=1, cat = LpInteger)
        fout = LpVariable.dicts('fout', index, lowBound=0, upBound=1, cat = LpInteger)
        f = LpVariable.dicts('f',index, lowBound=0, upBound=1, cat = LpInteger)
        fij = LpVariable.dicts('fij', (index, index), lowBound=0, upBound=1, cat = LpInteger)
        
        prob += lpSum([(fin[i] * c_in + fout[i] * c_out + f[i] * c_det(s_array[int(i)], v_det, v_link)) for i in index]) \
         + lpSum([lpSum([fij[i][j]*c_t(s_matrix[int(i)][int(j)], v_link) for j in index]) for i in index]), 'ObjectiveFunction'
        
        for i in index:
            prob += fin[i] + f[i] <= 1, ''
            prob += fout[i] + f[i] <= 1, ''
            tmp_index = list(index)
            tmp_index.remove(i)
            prob += fin[i] + f[i] - lpSum([fij[i][j] for j in tmp_index]) == 0, ''
            prob += fout[i] + f[i] - lpSum([fij[j][i] for j in tmp_index]) == 0, ''
        
            for j in index:
                #prob += fij[i][i] == 0, ''
                if i != j:
                    prob += fij[i][j] == fij[j][i], ''
        
                else:
                    prob += fij[i][i] == 0, ''
        
        
        #prob.writeLP("LinearProgram.lp")
        prob.solve()
        
        #print LpStatus[prob.status]
        
        #print fij
        
        
        #   Print the trajectory
        '''
        G = nx.DiGraph()
        G.add_nodes_from(range(0,n_test))
        g_pos=nx.spring_layout(G)

        for i in index:
            if fin[i].value() == 1: 
                nx.draw_networkx_nodes(G, g_pos, nodelist=[int(i)], edge_color = 'g', label = i)
            elif fout[i].value() == 1 and fin[i].value() != 1:
                nx.draw_networkx_nodes(G, g_pos, nodelist=[int(i)], edge_color = 'y', label = i)
            else:
                nx.draw_networkx_nodes(G, g_pos, nodelist=[int(i)], edge_color = 'r', label = i)        
        '''
        object_idx = 1
        for i in index:
            for j in index:
                #print fij[i][j].value()
                if (fij[i][j].value() == 1) & (i != j):
                   #G.add_edge(int(i), int(j))
                   #nx.draw_networkx_edges(G, g_pos, edgelist = [(int(i), int(j))], edge_color = 'b')
                   #print 'Edge: ' + i + ' ' + j
                   
                       
                   if ((int(i) < frame_1_locations_and_scores.shape[0]) and (int(j) >= frame_1_locations_and_scores.shape[0])):
                       if is_first_result:    
                           tmp_array = np.array(frame_1_locations_and_scores[int(i)])
                           tmp_array[1] = object_idx
                           result = np.array(tmp_array)
                           tmp_array = np.array(frame_2_locations_and_scores[int(j) - frame_1_locations_and_scores.shape[0]])
                           tmp_array[1] = object_idx
                           result = np.vstack((result,tmp_array))
                           object_idx += 1
                           is_first_result = False
                       else:
                           tmp_array = np.array(frame_1_locations_and_scores[int(i)])
                           tmp_array[1] = object_idx
                           result = np.vstack((result, tmp_array))
                           tmp_array = np.array(frame_2_locations_and_scores[int(j) - frame_1_locations_and_scores.shape[0]])
                           tmp_array[1] = object_idx
                           result = np.vstack((result, tmp_array))
                           object_idx += 1
                       print frame_1_locations_and_scores[int(i)], frame_2_locations_and_scores[int(j) - frame_1_locations_and_scores.shape[0]]
                   #if ((int(j) < frame_1_locations_and_scores.shape[0]) and (int(i) >= frame_1_locations_and_scores.shape[0])):
                   #    print frame_1_locations_and_scores[int(j)], frame_2_locations_and_scores[int(i) - frame_1_locations_and_scores.shape[0]]
        
        #nx.draw_networkx_labels(G, g_pos)
        #plt.show()
    #result1 = np.sort(result, axis = 0)
    a1 = result[:,::-1].T
    a2 = np.lexsort(a1)
    a3 = result[a2]
    np.savetxt(test_data_set[test_data_set_idx] + '.txt', a3, fmt='%f')
'''        
n_test = test_locations.shape[0] * 2
'''
'''
for i in range(0, pred.shape[0]):
    if pred[i, 0] > pred[i, 1]:
        pred_labels[i] = pred[i, 0]
    else:
        pred_labels[i] = pred[i, 1]
'''
'''
def save_data_as_hdf5_results(hdf5_data_filename, data, predict_labels):
    #HDF5 is one of the data formats Caffe accepts
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data.astype(np.float32)
        f['label'] = predict_labels.astype(np.float32)

save_data_as_hdf5_results('test_data_full_with_prediction_.hdf5', isolated_test_data_img[:, 0:3, ...], pred_labels) 
'''

'''
# test accuracy
pred = sg.predict(train_samples[n_train:])
pred_classes = [np.argmax(p) for p in pred]
#
_ = sg.evaluate(labels[n_train:], pred_classes)
'''