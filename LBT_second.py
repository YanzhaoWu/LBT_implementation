# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:10:55 2016

@author: v-yanzwu
"""

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.datasets import load_digits
#from stacked_generalizer import StackedGeneralizer
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import normalize

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
        print 'c_det ' + str(1 + (-s_i) / v_det)
        return  (1 + (-s_i) / v_det)
    else:
        print 'c_det ' + str(-1 + (-s_i + 1) / (1 - v_link))
        return (-1 + (-s_i + 1) / (1 - v_link))
        
        
def c_t(s_i_j, v_link):
    #print s_i_j
    if (s_i_j < v_link):
        print 'c_t ' + str(1 + (-s_i_j) / v_link)
        return (1 + (-s_i_j) / v_link)
    else:
        print 'c_t ' + str(-1 + (-s_i_j + 1) / (1 - v_link))
        return (-1 + (-s_i_j + 1) / (1 - v_link))

# Todo c_in = c_out, v_det and v_link
c_in = 0.2
c_out = 0.2

v_det = 0.2
v_link = 0.35




caffe.set_mode_cpu()
# Gradient Boost

#times = hdf5_file['timestamp'][:]
#train_data_img = hdf5_file['data'][:]

net = caffe.Net('LBT_deploy.prototxt', 'lbt__iter_45000.caffemodel', caffe.TEST)
#train_samples = np.zeros((labels.shape[0], 6))


# Generate Test Data
print 'Finish the fitting phase'
overlap_threshold = 0.8
def overlap_area(test_box, restrict_box):

    result = 0
    w = min(test_box[2] + test_box[0], restrict_box[2] + restrict_box[0]) - max(test_box[0], restrict_box[0]) + 1
    h = min(test_box[3] + test_box[1], restrict_box[3] + restrict_box[1]) - max(test_box[1], restrict_box[1]) + 1

    if w>0 and h>0:
        area = (test_box[2] + 1) * (test_box[3] + 1) + \
            (restrict_box[2] + 1) * (restrict_box[3] + 1) - w * h
        result = w * h / area;

    return result

def get_detection_locations_and_scores(frame_idx, detection_boxes):
    i = 0
    is_first_time = True
    result = None
    while (i < detection_boxes.shape[0]) and (detection_boxes[i][0] <= frame_idx) :
        if (detection_boxes[i][0] == frame_idx):
            if is_first_time:
                result = np.array([detection_boxes[i][0:7]])
                #print result
                is_first_time = False
            else:
                result = np.vstack((result, detection_boxes[i][0:7]))
                #print result
        i += 1
    print result
    return result            

def get_correct_object_idx(pre_frame_location, object_idx, result):
    i = result.shape[0] - 1
    while (i >= 0):
        #if (result[i][0] == pre_frame_location[0]) and (result[i][2] == pre_frame_location[2]) and (result[i][3] == pre_frame_location[3]) \
        #    and (result[i][2] == pre_frame_location[2]) and (result[i][3] == pre_frame_location[3]) :
        if (int(result[i][0]) == int(pre_frame_location[0]) or (int(result[i][0]) + 1) == int(pre_frame_location[0])) and (overlap_area(pre_frame_location[2:6], result[i][2:6]) > overlap_threshold):
            return result[i][1]
        i -= 1
    return (object_idx + 1)

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
    object_idx = 1
    for frame_idx in range(1, frame_num):
        frame_1_locations_and_scores = get_detection_locations_and_scores(frame_idx, detection_boxes)
        frame_2_locations_and_scores = get_detection_locations_and_scores(frame_idx + 1, detection_boxes)
        if frame_1_locations_and_scores == None or frame_2_locations_and_scores == None:
            continue
        
        n_test = frame_1_locations_and_scores.shape[0] + frame_2_locations_and_scores.shape[0]
        
        img1 = cv2.imread(test_data_path + test_data_set[test_data_set_idx] + '\\preprossed\\' + ('%06d' % frame_idx) + '.jpg')
        img2 = cv2.imread(test_data_path + test_data_set[test_data_set_idx] + '\\preprossed\\' + ('%06d' % (frame_idx + 1)) + '.jpg')
        optical_flow1 = cv2.imread(test_data_path + test_data_set[test_data_set_idx] + '\\optical_flow\\' + ('%06d' % frame_idx) + '.jpg')
        optical_flow2 = cv2.imread(test_data_path + test_data_set[test_data_set_idx] + '\\optical_flow\\' + ('%06d' % (frame_idx + 1)) + '.jpg')
        
        isolated_test_locations = np.zeros((n_test, 6))
        isolated_test_times = np.zeros(n_test)
        isolated_test_scores = np.zeros(n_test)
        isolated_test_data_img = np.zeros((n_test, 5, test_data_img_size[0], test_data_img_size[1]))
        
        counter = 0
        for idx in range(0, frame_1_locations_and_scores.shape[0]):
            isolated_test_times[counter] = frame_1_locations_and_scores[idx][0]
            isolated_test_locations[counter] = frame_1_locations_and_scores[idx][0:6]
            isolated_test_scores[counter] = frame_1_locations_and_scores[idx][6]
            
            tmp_img = img1[int(max(0, frame_1_locations_and_scores[idx][3])) : int(max(0, frame_1_locations_and_scores[idx][3]) + max(0, frame_1_locations_and_scores[idx][5] - 1)), int(max(0, frame_1_locations_and_scores[idx][2])) : int(max(0, frame_1_locations_and_scores[idx][2]) + max(0, frame_1_locations_and_scores[idx][4] - 1))]
            tmp_optical_flow = optical_flow1[int(max(0, frame_1_locations_and_scores[idx][3])) : int(max(0, frame_1_locations_and_scores[idx][3]) + max(0, frame_1_locations_and_scores[idx][5] - 1)), int(max(0, frame_1_locations_and_scores[idx][2])) : int(max(0, frame_1_locations_and_scores[idx][2]) + max(0, frame_1_locations_and_scores[idx][4] - 1))]
            for idx_channel in range(0, tmp_img.shape[2]):
                isolated_test_data_img[counter, idx_channel, ...] = cv2.resize(tmp_img[..., idx_channel], (test_data_img_size[1], test_data_img_size[0]))
            for idx_channel in range(tmp_img.shape[2], tmp_img.shape[2] + tmp_optical_flow.shape[2] - 1):
                isolated_test_data_img[counter, idx_channel, ...] = cv2.resize(tmp_optical_flow[..., idx_channel - tmp_img.shape[2] + 1], (test_data_img_size[1], test_data_img_size[0]))
            counter += 1
            
        for idx in range(0, frame_2_locations_and_scores.shape[0]):
            isolated_test_times[counter] = frame_2_locations_and_scores[idx][0]
            isolated_test_locations[counter] = frame_2_locations_and_scores[idx][0:6]
            isolated_test_scores[counter] = frame_2_locations_and_scores[idx][6]
            
            tmp_img = img2[int(max(0, frame_2_locations_and_scores[idx][3])) : int(max(0, frame_2_locations_and_scores[idx][3]) + max(0, frame_2_locations_and_scores[idx][5] - 1)), int(max(0, frame_2_locations_and_scores[idx][2])) : int(max(0, frame_2_locations_and_scores[idx][2]) + max(0, frame_2_locations_and_scores[idx][4] - 1))]
            tmp_optical_flow = optical_flow2[int(max(0, frame_2_locations_and_scores[idx][3])) : int(max(0, frame_2_locations_and_scores[idx][3]) + max(0, frame_2_locations_and_scores[idx][5] - 1)), int(max(0, frame_2_locations_and_scores[idx][2])) : int(max(0, frame_2_locations_and_scores[idx][2]) + max(0, frame_2_locations_and_scores[idx][4] - 1))]
            for idx_channel in range(0, tmp_img.shape[2]):
                isolated_test_data_img[counter, idx_channel, ...] = cv2.resize(tmp_img[..., idx_channel], (test_data_img_size[1], test_data_img_size[0]))
            for idx_channel in range(tmp_img.shape[2], tmp_img.shape[2] + tmp_optical_flow.shape[2] - 1):
                isolated_test_data_img[counter, idx_channel, ...] = cv2.resize(tmp_optical_flow[..., idx_channel - tmp_img.shape[2] + 1], (test_data_img_size[1], test_data_img_size[0]))
            counter += 1
        
        n_x = frame_1_locations_and_scores.shape[0]
        n_y = frame_2_locations_and_scores.shape[0]
        
        candidate_samples = np.zeros((n_x, n_y, 516), dtype = np.float32)
        
        for i in range(0, n_x):
            x0 = np.array([max(0, isolated_test_locations[i][2]) + 0.5 * max(0, isolated_test_locations[i][4]), max(0, isolated_test_locations[i][3]) + 0.5 * max(0, isolated_test_locations[i][5])]) # Center Location
            s0 = np.array([max(0, isolated_test_locations[i][4]), max(0, isolated_test_locations[i][5])])
            t0 = isolated_test_times[i]
            for j in range(n_x, n_x + n_y):
                x1 = np.array([max(0, isolated_test_locations[j][2]) + 0.5 * max(0, isolated_test_locations[j][4]), max(0, isolated_test_locations[j][3]) + 0.5 * max(0, isolated_test_locations[j][5])]) # Center Location
                s1 = np.array([max(0, isolated_test_locations[j][4]), max(0, isolated_test_locations[j][5])])    
                t1 = isolated_test_times[j]
            
                relative_size_change = (s0 - s1) / (s0 + s1)
                relative_velocity = (x0 - x1) / (t1 - t0)
                candidate_samples[i][j-n_x][0:2] = relative_size_change
                candidate_samples[i][j-n_x][2:4] = relative_velocity
                net.blobs['data'].data[0, 0:5, ...] = isolated_test_data_img[i]
                net.blobs['data'].data[0, 5:10, ...] = isolated_test_data_img[j]
                tmp_out = net.forward()
                cnn_prediction = tmp_out['fc6'].copy()
                #print cnn_prediction
                candidate_samples[i][j-n_x][4:516] = cnn_prediction.squeeze()
        
        candidate_samples[np.isfinite(candidate_samples) == False] = 10000
        candidate_samples[np.isnan(candidate_samples) == True] = 0
        candidate_samples = candidate_samples.reshape((candidate_samples.shape[0]*candidate_samples.shape[1], candidate_samples.shape[2]))
         
        pred = sg.predict(candidate_samples)
        pred_labels = np.zeros(pred.shape[0])
        pred_normalize = np.zeros(pred.shape)
        for i in range(0, pred.shape[0]):
            pred_normalize[i] = pred[i] / (pred[i][0] + pred[i][1])
        #pred_labels = pred[:, 0]
        pred_labels = pred_normalize[:, 1] # Predicted label == 1
        s_array = isolated_test_scores / 100
        #print s_array
        s_matrix = np.array(pred_labels).reshape((n_x, n_y))
        #print s_matrix
        

        index_x = ' '
        index_y = ' '
        index = ' '
        
        for x in range(0, n_test):
            index += str(x)
            index += ' '
        
        for i in range(0, n_x):
            index_x += str(i)
            index_x += ' '
        for j in range(0, n_y):
            index_y += str(j)
            index_y += ' '
        
        index = index.split()
        index_x = index_x.split()
        index_y = index_y.split()
        # raise NameError('Hi')
        prob = LpProblem('Binary Linear Programming', LpMinimize)
        fin = LpVariable.dicts('fin', index_x, lowBound=0, upBound=1, cat = LpInteger)
        fout = LpVariable.dicts('fout', index_y, lowBound=0, upBound=1, cat = LpInteger)
        f = LpVariable.dicts('f',index, lowBound=0, upBound=1, cat = LpInteger)
        fij = LpVariable.dicts('fij', (index_x, index_y), lowBound=0, upBound=1, cat = LpInteger)
        
        prob += lpSum([fin[i] * c_in for i in index_x]) \
         + lpSum([fout[i] * c_out for i in index_y]) \
         + lpSum([f[i] * c_det(s_array[int(i)], v_det, v_link) for i in index]) \
         + lpSum([lpSum([fij[i][j]*c_t(s_matrix[int(i)][int(j)], v_link) for j in index_y]) for i in index_x]), 'ObjectiveFunction'
        
        for i in index_x:
            prob += fin[i] + f[i] <= 1, ''
            prob += fin[i] + f[i] - lpSum([fij[i][j] for j in index_y]) == 0, ''
        
        for i in index_y:    
            prob += fout[i] + f[i] - lpSum([fij[j][i] for j in index_x]) == 0, ''
            prob += fout[i] + f[i] <= 1, ''
        
        
        #prob.writeLP("LinearProgram.lp")
        prob.solve()
        
        #print LpStatus[prob.status]
        
        #print fij
        
        
        #   Print the trajectory
        #G = nx.DiGraph()
        #G.add_nodes_from(range(0,n_test))
        #g_pos=nx.spring_layout(G)
        '''
        for i in index:
            if int(i) < n_x and fin[i].value() == 1: 
                nx.draw_networkx_nodes(G, g_pos, nodelist=[int(i)], node_color = 'g', label = i)
            elif int(i) >= n_x and fout[str(int(i) - n_x)].value() == 1:
                nx.draw_networkx_nodes(G, g_pos, nodelist=[int(i)], node_color = 'y', label = i)
            else:
                nx.draw_networkx_nodes(G, g_pos, nodelist=[int(i)], node_color = 'r', label = i)
        '''
        for i in index_x:
            for j in index_y:
                #print fij[i][j].value()
                if (fij[i][j].value() == 1):
                   #G.add_edge(int(i), int(j) + n_x)
                   #nx.draw_networkx_edges(G, g_pos, edgelist = [(int(i), int(j) + n_x)], edge_color = 'b')
                   print 'Edge: ' + i + ' ' + str(int(j) + n_y)
                   #if ((int(i) < frame_1_locations_and_scores.shape[0]) and (int(j) >= frame_1_locations_and_scores.shape[0])):
                   if is_first_result:    
                       tmp_array = np.array(frame_1_locations_and_scores[int(i)])
                       tmp_array[1] = object_idx
                       result = np.array([tmp_array])
                       tmp_array = np.array(frame_2_locations_and_scores[int(j)])
                       tmp_array[1] = object_idx
                       result = np.vstack((result,tmp_array))
                       is_first_result = False
                   else:
                       tmp_array = np.array(frame_1_locations_and_scores[int(i)])
                       tmp_object_idx = get_correct_object_idx(frame_1_locations_and_scores[int(i)], object_idx, result)
                       if tmp_object_idx > object_idx:
                           object_idx = tmp_object_idx
                           tmp_array[1] = tmp_object_idx
                           result = np.vstack((result, tmp_array))
                           
                       tmp_array = np.array(frame_2_locations_and_scores[int(j)])
                       tmp_array[1] = tmp_object_idx
                       result = np.vstack((result, tmp_array))
                   print frame_1_locations_and_scores[int(i)], frame_2_locations_and_scores[int(j)]

                   #if ((int(j) < frame_1_locations_and_scores.shape[0]) and (int(i) >= frame_1_locations_and_scores.shape[0])):
                   #    print frame_1_locations_and_scores[int(j)], frame_2_locations_and_scores[int(i) - frame_1_locations_and_scores.shape[0]]
        
        #nx.draw_networkx_labels(G, g_pos)
        #plt.show()
    #result1 = np.sort(result, axis = 0)
    a1 = result[:,::-1].T
    a2 = np.lexsort(a1)
    a3 = result[a2]
    np.savetxt(test_data_set[test_data_set_idx] + 'small.txt', a3, fmt='%f')
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