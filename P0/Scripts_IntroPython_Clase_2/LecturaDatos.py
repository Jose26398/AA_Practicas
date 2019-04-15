# -*- coding: utf-8 -*-
import numpy as np

matrix = []

f = open('mat.txt','r')

for l in f:
    row_matrix = []
    l = l.rstrip()
    
    for e in l.split(' '):
        #if e != '':
        row_matrix.append(float(e))
            
    if len(row_matrix) > 0:
        matrix.append(row_matrix)
    
f.close()
    
matrix = np.array(matrix, np.float64)

