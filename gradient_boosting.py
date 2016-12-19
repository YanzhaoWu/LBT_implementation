# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 21:15:49 2016

@author: v-yanzwu
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from stacked_generalizer import StackedGeneralizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Gradient Boost
x0 = np.array([0,0])
x1 = np.array([2,2])
s0 = np.array([1,1])
s1 = np.array([1,1])
t0 = 0
t1 = 1

relative_size_change = (s0 - s1) / (s0 + s1)
relative_velocity = (x0 - x1) / (t1 - t0)

cnn_prediction = np.zeros(512)


VERBOSE = True
N_FOLDS = 5

# load data and shuffle observations
# data = load_digits()

#X = data.data # Change to another type
#y = data.target
print relative_size_change.shape
print relative_velocity.shape
print cnn_prediction.shape

#test_sample = [relative_size_change, relative_velocity, cnn_prediction]
#train_sample = np.array(test_sample)

test_sample = np.append(np.append(relative_size_change, relative_velocity), cnn_prediction)
print test_sample.shape

train_sample = np.random.randint(0, 255,(100,test_sample.shape[0]))

# print test_sample.shape
# print train_sample

#X = np.random.randint(0,255, test_sample.shape)
y = np.random.randint(0,2, train_sample.shape[0])

#print y.shape
#print y

shuffle_idx = np.random.permutation(y.shape[0])

X = train_sample[shuffle_idx]
y = y[shuffle_idx]

# hold out 20 percent of data for testing accuracy
train_prct = 0.8
n_train = int(round(X.shape[0]*train_prct))

# define base models
base_models = [GradientBoostingClassifier(n_estimators=100),
               GradientBoostingClassifier(n_estimators=100),
               GradientBoostingClassifier(n_estimators=100)]

# define blending model
blending_model = LogisticRegression()

# initialize multi-stage model
sg = StackedGeneralizer(base_models, blending_model, 
                        n_folds=N_FOLDS, verbose=VERBOSE)

# fit model
sg.fit(X[:n_train],y[:n_train])

# test accuracy
pred = sg.predict(X[n_train:])
pred_classes = [np.argmax(p) for p in pred]

_ = sg.evaluate(y[n_train:], pred_classes)
