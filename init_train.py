# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:00:53 2016

@author: v-yanzwu
"""

train_data_path = 'D:\\SharedData\\LBT_Train_Set\\train-210'
train_data_prefix = 'IFWN-sequence-'
trian_data_middle = '.avi-'
train_data_suffix = '-R200.png'



scan_files()

for i in range(89, ):
   
    
    print  train_data_prefix + ('%05d' % (i*7 + 89)) + train_data_suffix

