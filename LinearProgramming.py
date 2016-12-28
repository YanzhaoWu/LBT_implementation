# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 22:36:04 2016

@author: v-yanzwu
"""
import numpy as np
# Tracking with Linear Programming
import h5py
import cv2
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

# manual input


hdf5_data_path = 'test_data_full_with_prediction_.hdf5'
hdf5_file = h5py.File(hdf5_data_path, 'r')
prediction = hdf5_file['label'][:]
test_img = hdf5_file['data'][:]
# matrix solution
n = test_img.shape[0]
length = (3 + n) * n

s_array = np.random.randint(0,2,n)
#s_array = np.ones(n)
print s_array
#raise NameError('Stop')
s_matrix = np.random.randint(0,2,(n,n))#
#s_matrix = np.array(prediction).reshape((n,n))
print s_matrix
'''
# coefficient

coefficient = np.zeros(length)
for i in range(0,n):
    coefficient[i] = c_in
    coefficient[i + n] = c_out
    coefficient[i + n * 2] = c_det(s_array[i], v_det, v_link)
    for j in range(0, n):
        coefficient[i * n + n * 3 + j] = c_t(s_matrix[i][j], v_link)

print coefficient

# restrict1
restrict1_right = np.ones(2 * n)
restrict1_left = np.zeros((2 * n, length))

for i in range(0, n):
    restrict1_left[i][i] = 1
    restrict1_left[i][i + n * 2] = 1
    restrict1_left[i + n][i + n] = 1
    restrict1_left[i + n][i + n * 2] = 1
print restrict1_left
# restrict1

restrict2_right = np.zeros(2 * n)
restrict2_left = np.zeros((2 * n, length))

for i in range(0, n):
    restrict2_left[i][i] = 1
    restrict2_left[i][i + n * 2] = 1
    restrict2_left[i + n][i + n] = 1
    restrict2_left[i + n][i + n * 2] = 1
    for j in range(0, n):
        restrict2_left[i][n * 3 + i * n + j] = -1
        restrict2_left[i + n][n * 3 + i * n + j] = -1
'''
from pulp import LpProblem, LpVariable, LpInteger, LpMinimize, lpSum, LpStatus

index = ' '

for i in range(0,n):
    index += str(i)
    index += ' '

index = index.split()
#print index


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


prob.writeLP("LinearProgram.lp")
prob.solve()

print LpStatus[prob.status]

#print fij


#   Print the trajectory

import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()
G.add_nodes_from(range(0,n))
g_pos=nx.spring_layout(G)


for i in index:
    for j in index:
        #print fij[i][j].value()
        if (fij[i][j].value() == 1) & (i != j):
           G.add_edge(int(i), int(j))
           nx.draw_networkx_edges(G, g_pos, edgelist = [(int(i), int(j))], edge_color = 'b')
           print 'Edge: ' + i + ' ' + j

           
for i in index:
    if fin[i].value() == 1: 
        nx.draw_networkx_nodes(G, g_pos, nodelist=[int(i)], edge_color = 'g', label = i)
    elif fout[i].value() == 1 and fin[i].value() != 1:
        nx.draw_networkx_nodes(G, g_pos, nodelist=[int(i)], edge_color = 'y', label = i)
    else:
        nx.draw_networkx_nodes(G, g_pos, nodelist=[int(i)], edge_color = 'r', label = i)
print fin, fout
nx.draw_networkx_labels(G, g_pos)

contrast = np.transpose(test_img[0])
for i in range(1, n):
    contrast = np.hstack((contrast, np.transpose(test_img[i])))
plt.show()
cv2.imwrite('test.png', contrast)
