# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:50:25 2016

@author: v-yanzwu
"""
import caffe
solver_prototxt_filename = 'LBT_solver.prototxt'

def train(solver_prototxt_filename):
    '''
    Train the CNN
    '''
    caffe.set_mode_cpu()
    solver = caffe.get_solver(solver_prototxt_filename)
    solver.solve()
    
if __name__ == '__main__':
    #net = caffe.Net('LBT_train_test.prototxt', caffe.TRAIN)
    train(solver_prototxt_filename)