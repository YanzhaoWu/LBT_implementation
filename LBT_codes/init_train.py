# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:00:53 2016

@author: v-yanzwu
"""


'''
import os
def walk_dir(dir,fileinfo,topdown=True):
for root, dirs, files in os.walk(dir, topdown):
for name in files:
print(os.path.join(name))
fileinfo.write(os.path.join(root,name)+'\n')
for name in dirs:
print(os.path.join(name))
fileinfo.write(' '+ os.path.join(root,name)+'\n')
dir = raw_input('please input the path:')
dir = r'G:\codes\python\file_info'
fileinfo = open('list2.txt','w')
walk_dir(dir,fileinfo)
fileinfo.close()
'''


train_data_path = 'D:\\SharedData\\LBT_Train_Set\\train-210\\'
train_data_prefix = 'IFWN-sequence-'
train_data_middle = '.avi-'
train_data_suffix = '-R200.png'
train_data_file_suffix = 'png'
train_data_coordinate_file = 'train-210.idl'
train_data_group_id_digit = 3
train_data_time_id_digit = 5
train_data_optical_flow_path = 'D:\SharedData\LBT_Train_Set\train-210\maps'
train_data_optical_flow_suffix = '-R200-map.png'

#print  train_data_prefix + ('%05d' % (i*7 + 89)) + train_data_suffix

import os
import cv2
import matplotlib.pyplot as plt

def get_curr_dir_files(path, file_suffix = None):
    result_files = []
    is_file_filter = (file_suffix != None)
    for root, dirs, files in os.walk(path):
        for files_path in files:
            file_path = os.path.join(root, files_path)
            extension = os.path.splitext(file_path)[1][1:]
            #print extension
            if is_file_filter and extension in file_suffix and extension != '': # README no extension!
                result_files.append(files_path)
            elif not is_file_filter:
                result_files.append(files_path)
    return result_files


files = get_curr_dir_files(train_data_path, train_data_file_suffix)
coordinate_file = open(train_data_path + train_data_coordinate_file, 'r')
coordinate_file_lines = coordinate_file.readlines()
#print coordinate_file_lines
import re

coordinates = dict()
for line in coordinate_file_lines:
    line_parts = line.split(':')
    #print line_parts
    coordinates[line_parts[0]] =[int(num) for num in re.findall('\d+', line_parts[1])]

#print coordinates
img_group = dict()
organized_train_data_group = dict()
#print organized_train_data_group
train_data_prefix_len = len(train_data_prefix)
train_data_time_prefix_len = train_data_prefix_len + train_data_group_id_digit + len(train_data_middle)

#print files

for file_name in files:
    #print file_name[train_data_prefix_len : train_data_prefix_len + train_data_group_id_digit]
    img_group[file_name[train_data_prefix_len : train_data_prefix_len + train_data_group_id_digit]] = list()
    organized_train_data_group[file_name[train_data_prefix_len : train_data_prefix_len + train_data_group_id_digit]] = list()
for file_name in files:
    #print file_name[train_data_time_prefix_len : train_data_time_prefix_len + train_data_time_id_digit]
    img_group[file_name[train_data_prefix_len : train_data_prefix_len + 
                        train_data_group_id_digit]].append(file_name[train_data_time_prefix_len : train_data_time_prefix_len + 
                        train_data_time_id_digit])

#print img_group
#print organized_train_data_group
from img_preprocess import *
from optical_flow_generator import *
#img_size = (320, 400) 
#for img_group_index in img_group:
#    #print len(img_group[img_group_index]), img_group[img_group_index]
#    tmp_list = img_group[img_group_index]
#    #print len(tmp_list)
#    tmp_data_list = list()
#    tmp_img = np.zeros((320,400,5))
#    #print tmp_img.shape
#    for idx in range(0, len(tmp_list) - 1):
#        img_1 = cv2.imread(train_data_path + train_data_prefix + img_group_index + train_data_middle + tmp_list[idx] + train_data_suffix)
#        img_1_resized = cv2.resize(img_1, (img_size[1], img_size[0]))
#        tmp_img[:, :, 0:3] = img_1_resized 
#        #print img_1.shape
#        #plt.imshow(img_1)
#        #print train_data_path + train_data_prefix + img_group_index + train_data_middle + tmp_list[idx] + train_data_suffix
#        img_2 = cv2.imread(train_data_path + train_data_prefix + img_group_index + train_data_middle + tmp_list[idx + 1] + train_data_suffix)
#        tmp_img_1 = img_preprocess_func(img_1)
#        tmp_optical_flow = generate_optical_flow(img_1, img_2, img_1.shape)
#        tmp_optical_flow_resized = cv2.resize(tmp_optical_flow, (img_size[1], img_size[0]))
#        tmp_img[:, :, 3:5] = tmp_optical_flow_resized[:, :, 1:3]
#        organized_train_data_group[img_group_index].append(tmp_img)

#img_size = (320, 400)
for img_group_index in img_group:
    #print len(img_group[img_group_index]), img_group[img_group_index]
    tmp_list = img_group[img_group_index]
    #print len(tmp_list)
    tmp_data_list = list()
    #print tmp_img.shape
    for idx in range(0, len(tmp_list) - 1):
        img_1 = cv2.imread(train_data_path + train_data_prefix + img_group_index + train_data_middle + tmp_list[idx] + train_data_suffix)
        img_1_resized = img_1#= cv2.resize(img_1, (img_size[1], img_size[0]))
        img_size = img_1.shape[0:2]
        tmp_img = np.zeros((img_size[0],img_size[1],5), dtype = img_1.dtype)
        tmp_img[:, :, 0:3] = img_1_resized 
        #print img_1.shape
        #plt.imshow(img_1)
        #print train_data_path + train_data_prefix + img_group_index + train_data_middle + tmp_list[idx] + train_data_suffix
        img_2 = cv2.imread(train_data_path + train_data_prefix + img_group_index + train_data_middle + tmp_list[idx + 1] + train_data_suffix)
        tmp_img_1 = img_preprocess_func(img_1)
        tmp_optical_flow = generate_optical_flow(img_1, img_2, img_1.shape)
        tmp_optical_flow_resized = cv2.resize(tmp_optical_flow, (img_size[1], img_size[0]))
        tmp_img[:, :, 3:5] = tmp_optical_flow_resized[:, :, 1:3]
        tmp_coordinate = coordinates['\"' + train_data_prefix + img_group_index + train_data_middle + tmp_list[idx] + train_data_suffix + '\"']
        print tmp_coordinate
        tmp_img_fragment = tmp_img[tmp_coordinate[1] : tmp_coordinate[3], tmp_coordinate[0] : tmp_coordinate[2], ...]
        organized_train_data_group[img_group_index].append(tmp_img_fragment)
        
# Generate the train data
import caffe

#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2,0,1)) # eg, from (227,227,3) to (3,227,227)
#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

num_correct_data = 100
num_wrong_data = 100
train_data_labels = np.zeros(num_correct_data + num_wrong_data)
train_data_img_size = (121, 53, 10)
train_data_img = np.zeros((num_correct_data + num_wrong_data, train_data_img_size[2], train_data_img_size[0], train_data_img_size[1]))
train_data_location = np.zeros((num_correct_data + num_wrong_data, 2, 4))
train_data_time_stamp = np.zeros((num_correct_data + num_wrong_data, 2))
# Correct 
counter = 0
for img_group_index in img_group:
    tmp_list = img_group[img_group_index]
    
    for idx in range(0, len(tmp_list) - 2):
        img_1 = organized_train_data_group[img_group_index][idx]
        img_2 = organized_train_data_group[img_group_index][idx + 1]
        train_data_location[counter, 0, ...] = coordinates['\"' + train_data_prefix + img_group_index + train_data_middle + tmp_list[idx] + train_data_suffix + '\"']
        train_data_location[counter, 1, ...] = coordinates['\"' + train_data_prefix + img_group_index + train_data_middle + tmp_list[idx + 1] + train_data_suffix + '\"']
        train_data_time_stamp[counter][0] = idx
        train_data_time_stamp[counter][1] = idx + 1

        tmp_img = np.zeros((train_data_img_size[2], train_data_img_size[0], train_data_img_size[1]), dtype = img_1.dtype)
#        tmp_img[0:3, ...] = cv2.resize(img_1[:, :, 0:3], (train_data_img_size[1], train_data_img_size[0]))
#        tmp_img[5:8, ...] = cv2.resize(img_2[:, :, 0:3], (train_data_img_size[1], train_data_img_size[0]))
#        
#        v_img = np.zeros((img_1.shape[0], img_1.shape[1],3))
#        v_img[:, :, 0:2] = img_1[:, :, 3:5]
#        tmp_img[3:5, ...] = cv2.resize(v_img, (train_data_img_size[1], train_data_img_size[0]))
#        
#        v_img = np.zeros((img_2.shape[0], img_2.shape[1],3))
#        v_img[:, :, 0:2] = img_2[:, :, 3:5]
#        tmp_img[8:10, ...] = cv2.resize(v_img, (train_data_img_size[1], train_data_img_size[0])) 
        
        tmp_img_1 = cv2.resize(img_1, (train_data_img_size[1], train_data_img_size[0]))
        tmp_img_2 = cv2.resize(img_2, (train_data_img_size[1], train_data_img_size[0]))
        for i in range(0, tmp_img_1.shape[2]):
            tmp_img[i, ...] = tmp_img_1[..., i]
        for i in range(tmp_img_1.shape[2], tmp_img_1.shape[2] + tmp_img_2.shape[2]):
            tmp_img[i, ...] = tmp_img_2[..., i - tmp_img_1.shape[2]] 

        train_data_img[counter, ...] = tmp_img
        train_data_labels[counter] = 1
        counter += 1
        if counter >= num_correct_data:
            break
    if counter >= num_correct_data:
        break
# Wrong 

img_group_index_array = list()
for img_group_index in img_group:
    img_group_index_array.append(img_group_index)

import random
counter = 0
while counter < num_wrong_data:
    random_1 = random.randint(0, len(img_group_index_array) - 1)
    random_2 = random.randint(0, len(img_group_index_array) - 1)
    if random_1 == random_2:
        continue
    tmp_list_1 = img_group[img_group_index_array[random_1]]
    tmp_list_2 = img_group[img_group_index_array[random_2]]
    random_1_index = random.randint(0, len(tmp_list_1) - 2)
    random_2_index = random.randint(0, len(tmp_list_2) - 2)
    img_1 = organized_train_data_group[img_group_index_array[random_1]][random_1_index]
    img_2 = organized_train_data_group[img_group_index_array[random_2]][random_2_index]
    train_data_location[counter + num_correct_data, 0, ...] = coordinates['\"' + train_data_prefix + img_group_index_array[random_1] + train_data_middle + img_group[img_group_index_array[random_1]][random_1_index] + train_data_suffix + '\"']
    train_data_location[counter + num_correct_data, 1, ...] = coordinates['\"' + train_data_prefix + img_group_index_array[random_2] + train_data_middle + img_group[img_group_index_array[random_2]][random_2_index] + train_data_suffix + '\"']
    train_data_time_stamp[counter + num_correct_data][0] = random_1_index
    train_data_time_stamp[counter + num_correct_data][1] = random_2_index
    tmp_img = np.zeros((train_data_img_size[2], train_data_img_size[0], train_data_img_size[1]), dtype = img_1.dtype)
#        tmp_img[0:3, ...] = cv2.resize(img_1[:, :, 0:3], (train_data_img_size[1], train_data_img_size[0]))
#        tmp_img[5:8, ...] = cv2.resize(img_2[:, :, 0:3], (train_data_img_size[1], train_data_img_size[0]))
#        
#        v_img = np.zeros((img_1.shape[0], img_1.shape[1],3))
#        v_img[:, :, 0:2] = img_1[:, :, 3:5]
#        tmp_img[3:5, ...] = cv2.resize(v_img, (train_data_img_size[1], train_data_img_size[0]))
#        
#        v_img = np.zeros((img_2.shape[0], img_2.shape[1],3))
#        v_img[:, :, 0:2] = img_2[:, :, 3:5]
#        tmp_img[8:10, ...] = cv2.resize(v_img, (train_data_img_size[1], train_data_img_size[0])) 
        
    tmp_img_1 = cv2.resize(img_1, (train_data_img_size[1], train_data_img_size[0]))
    tmp_img_2 = cv2.resize(img_2, (train_data_img_size[1], train_data_img_size[0]))
    for i in range(0, tmp_img_1.shape[2]):
        tmp_img[i, ...] = tmp_img_1[..., i]
    for i in range(tmp_img_1.shape[2], tmp_img_1.shape[2] + tmp_img_2.shape[2]):
        tmp_img[i, ...] = tmp_img_2[..., i - tmp_img_1.shape[2]] 

    train_data_img[counter + num_correct_data, ...] = tmp_img
    train_data_labels[counter + num_correct_data] = 0
    counter += 1

import h5py
def save_data_as_hdf5(hdf5_data_filename, data, label, location, time_stamp):
    '''
    HDF5 is one of the data formats Caffe accepts
    '''
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data.astype(np.float32)
        f['label'] = label.astype(np.float32)
        f['location'] = location.astype(np.uint8)
        f['timestamp'] = time_stamp.astype(np.uint8)

save_data_as_hdf5('train_data_full_.hdf5', train_data_img, train_data_labels, train_data_location, train_data_time_stamp)