# -*- coding: utf-8 -*-

import numpy as np
ini = 1
end = 4
x = np.tile(np.arange(ini, end+1), (end+1,1))
x

y = x.copy()
for i in range(x.shape[0]): 
    for j in range(x.shape[1]):
        x[i,j] **= 2
x
        
y **= 2 #Equivalent to previous two nested for-loops
y        
  
for i in range(end-1, end+1): 
    for j in range(x.shape[1]):
        x[i, j] += 5
x
        
y[np.arange(end-1,end+1), :] += 5 #Equivalent to previous two nested for-loops
y
